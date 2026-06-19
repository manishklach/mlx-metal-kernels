// Body-only MLX custom Metal kernel source.
// MLX generates the `kernel void fast_attention_tiled_kv_forward(...)`
// signature from input_names=["Q", "K", "V", "meta", "scale"] and
// output_names=["O"].
//
// Experimental tiled-K/V implementation:
//   - one threadgroup computes one attention row O[b, q, h, :]
//   - K/V are staged into threadgroup memory in KV_TILE-sized blocks
//   - the row streams over blocks using online softmax state (m, l, o)
//   - output accumulation is split across D dimensions
//
// Notes:
//   - This favors correctness and a clean online-softmax structure.
//   - It avoids materializing the full [S, S] attention matrix.

const uint tid = thread_position_in_threadgroup.x;
const uint row_id = threadgroup_position_in_grid.x;
const uint tg_size = threads_per_threadgroup.x;

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

threadgroup float q_tile[MAX_HEAD_DIM];
threadgroup float k_tile[KV_TILE * MAX_HEAD_DIM];
threadgroup float v_tile[KV_TILE * MAX_HEAD_DIM];
threadgroup float tile_scores[KV_TILE];
threadgroup float shared_m;
threadgroup float shared_l;
threadgroup float shared_m_new;
threadgroup float shared_alpha;
threadgroup float shared_tile_sum;

for (uint d = tid; d < uint(D); d += tg_size) {
    q_tile[d] = float(Q[q_base + int(d)]);
}

if (tid == 0) {
    shared_m = -INFINITY;
    shared_l = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const uint owned_dims = (tid < uint(D)) ? ((uint(D) - tid + tg_size - 1u) / tg_size) : 0u;
float acc0 = 0.0f;
float acc1 = 0.0f;

for (int tile_start = 0; tile_start < S; tile_start += KV_TILE) {
    const int tile_len = min(KV_TILE, S - tile_start);

    for (uint idx = tid; idx < uint(tile_len * D); idx += tg_size) {
        const int tk = int(idx / uint(D));
        const int d = int(idx % uint(D));
        const int global_k = tile_start + tk;
        const int base = b * stride_b + global_k * stride_s + h * stride_h;
        k_tile[tk * MAX_HEAD_DIM + d] = float(K[base + d]);
        v_tile[tk * MAX_HEAD_DIM + d] = float(V[base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float tile_max = -INFINITY;
        for (int tk = 0; tk < tile_len; ++tk) {
            const int global_k = tile_start + tk;
            if (causal && global_k > q) {
                tile_scores[tk] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q_tile[d] * k_tile[tk * MAX_HEAD_DIM + d];
            }

            const float score = dot * scale_val;
            tile_scores[tk] = score;
            tile_max = max(tile_max, score);
        }

        shared_m_new = max(shared_m, tile_max);
        shared_alpha = isfinite(shared_m) ? exp(shared_m - shared_m_new) : 0.0f;

        float tile_sum = 0.0f;
        for (int tk = 0; tk < tile_len; ++tk) {
            if (isfinite(tile_scores[tk])) {
                tile_sum += exp(tile_scores[tk] - shared_m_new);
            }
        }
        shared_tile_sum = tile_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (owned_dims >= 1u) {
        acc0 *= shared_alpha;
    }
    if (owned_dims >= 2u) {
        acc1 *= shared_alpha;
    }

    for (int tk = 0; tk < tile_len; ++tk) {
        if (!isfinite(tile_scores[tk])) {
            continue;
        }
        const float weight = exp(tile_scores[tk] - shared_m_new);
        const int tile_base = tk * MAX_HEAD_DIM;
        if (owned_dims >= 1u) {
            acc0 += weight * v_tile[tile_base + int(tid)];
        }
        if (owned_dims >= 2u) {
            acc1 += weight * v_tile[tile_base + int(tid + tg_size)];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        shared_l = shared_l * shared_alpha + shared_tile_sum;
        shared_m = shared_m_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

const float inv_l = 1.0f / max(shared_l, 1.0e-20f);
if (owned_dims >= 1u) {
    O[q_base + int(tid)] = ELEM_TYPE(acc0 * inv_l);
}
if (owned_dims >= 2u) {
    O[q_base + int(tid + tg_size)] = ELEM_TYPE(acc1 * inv_l);
}
