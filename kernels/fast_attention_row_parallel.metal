// Body-only MLX custom Metal kernel source.
// MLX generates the `kernel void fast_attention_row_parallel_forward(...)`
// signature from input_names=["Q", "K", "V", "meta", "scale"] and
// output_names=["O"].
//
// Experimental row-parallel implementation:
//   - one threadgroup computes one attention row O[b, q, h, :]
//   - threads cooperatively compute the Q·K dot product for each key position
//   - row max and denominator use threadgroup reductions
//   - output accumulation is parallelized across D dimensions
//
// Notes:
//   - This keeps the existing streaming three-pass structure for correctness.
//   - It intentionally avoids a full tiled/threadgroup-memory FlashAttention
//     implementation for now; that remains the next roadmap stage.

const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simd_id = simdgroup_index_in_threadgroup;
const uint tg_size = threads_per_threadgroup.x;
const uint num_simdgroups = (tg_size + 31u) / 32u;
const uint row_id = threadgroup_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];
const bool causal = (meta[4] != 0);

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * S * H);
if (row_id >= total_rows) {
    return;
}

const int h = int(row_id % uint(H));
const int q = int((row_id / uint(H)) % uint(S));
const int b = int(row_id / uint(S * H));

const int stride_b = S * H * D;
const int stride_s = H * D;
const int stride_h = D;
const int q_base = b * stride_b + q * stride_s + h * stride_h;
const float scale_val = scale[0];

threadgroup float simd_partials[32];
threadgroup float shared_score;
threadgroup float shared_row_max;
threadgroup float shared_denom;

if (tid == 0) {
    shared_row_max = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Pass 1: row max.
for (int k_idx = 0; k_idx < S; ++k_idx) {
    float partial_dot = 0.0f;
    if (!(causal && k_idx > q)) {
        const int k_base = b * stride_b + k_idx * stride_s + h * stride_h;
        for (uint d = tid; d < uint(D); d += tg_size) {
            partial_dot += float(Q[q_base + int(d)]) * float(K[k_base + int(d)]);
        }
    }

    float reduced = simd_sum(partial_dot);
    if (lane == 0) {
        simd_partials[simd_id] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        float block_sum = (tid < num_simdgroups) ? simd_partials[tid] : 0.0f;
        block_sum = simd_sum(block_sum);
        if (tid == 0 && !(causal && k_idx > q)) {
            shared_row_max = max(shared_row_max, block_sum * scale_val);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    shared_denom = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Pass 2: denominator.
for (int k_idx = 0; k_idx < S; ++k_idx) {
    float partial_dot = 0.0f;
    if (!(causal && k_idx > q)) {
        const int k_base = b * stride_b + k_idx * stride_s + h * stride_h;
        for (uint d = tid; d < uint(D); d += tg_size) {
            partial_dot += float(Q[q_base + int(d)]) * float(K[k_base + int(d)]);
        }
    }

    float reduced = simd_sum(partial_dot);
    if (lane == 0) {
        simd_partials[simd_id] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        float block_sum = (tid < num_simdgroups) ? simd_partials[tid] : 0.0f;
        block_sum = simd_sum(block_sum);
        if (tid == 0) {
            shared_score = block_sum * scale_val;
            if (!(causal && k_idx > q)) {
                shared_denom += exp(shared_score - shared_row_max);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 3: weighted V accumulation, parallelized over D.
const uint owned_dims = (tid < uint(D)) ? ((uint(D) - tid + tg_size - 1u) / tg_size) : 0u;
float acc0 = 0.0f;
float acc1 = 0.0f;

for (int k_idx = 0; k_idx < S; ++k_idx) {
    float partial_dot = 0.0f;
    if (!(causal && k_idx > q)) {
        const int k_base = b * stride_b + k_idx * stride_s + h * stride_h;
        for (uint dd = tid; dd < uint(D); dd += tg_size) {
            partial_dot += float(Q[q_base + int(dd)]) * float(K[k_base + int(dd)]);
        }
    }

    float reduced = simd_sum(partial_dot);
    if (lane == 0) {
        simd_partials[simd_id] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        float block_sum = (tid < num_simdgroups) ? simd_partials[tid] : 0.0f;
        block_sum = simd_sum(block_sum);
        if (tid == 0) {
            shared_score = block_sum * scale_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!(causal && k_idx > q)) {
        const int v_base = b * stride_b + k_idx * stride_s + h * stride_h;
        const float p = exp(shared_score - shared_row_max) / max(shared_denom, 1.0e-20f);
        if (owned_dims >= 1u) {
            acc0 += p * float(V[v_base + int(tid)]);
        }
        if (owned_dims >= 2u) {
            acc1 += p * float(V[v_base + int(tid + tg_size)]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (owned_dims >= 1u) {
    O[q_base + int(tid)] = ELEM_TYPE(acc0);
}
if (owned_dims >= 2u) {
    O[q_base + int(tid + tg_size)] = ELEM_TYPE(acc1);
}
