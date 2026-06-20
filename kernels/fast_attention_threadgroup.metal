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
const int q_idx = int((row_id / uint(H)) % uint(S));
const int b = int(row_id / uint(S * H));

const int stride_b = S * H * D;
const int stride_s = H * D;
const int stride_h = D;
const int q_base = b * stride_b + q_idx * stride_s + h * stride_h;
const float scale_val = scale[0];

threadgroup float q_shared[MAX_HEAD_DIM];
threadgroup float scratch[TG_THREADS];
threadgroup float shared_score;
threadgroup float shared_m;
threadgroup float shared_l;
threadgroup float shared_alpha;
threadgroup float shared_beta;

for (uint d = tid; d < uint(D); d += tg_size) {
    q_shared[d] = float(Q[q_base + int(d)]);
}

if (tid == 0) {
    shared_m = -INFINITY;
    shared_l = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float acc = 0.0f;
const bool owns_dim = (tid < uint(D));

for (int k_idx = 0; k_idx < S; ++k_idx) {
    if (causal && k_idx > q_idx) {
        continue;
    }

    const int kv_base = b * stride_b + k_idx * stride_s + h * stride_h;
    float partial_dot = 0.0f;
    for (uint d = tid; d < uint(D); d += tg_size) {
        partial_dot += q_shared[d] * float(K[kv_base + int(d)]);
    }
    scratch[tid] = partial_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint reduce = tg_size / 2u; reduce > 0u; reduce >>= 1u) {
        if (tid < reduce) {
            scratch[tid] += scratch[tid + reduce];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        shared_score = scratch[0] * scale_val;
        const float m_new = max(shared_m, shared_score);
        shared_alpha = isfinite(shared_m) ? exp(shared_m - m_new) : 0.0f;
        shared_beta = exp(shared_score - m_new);
        shared_l = shared_l * shared_alpha + shared_beta;
        shared_m = m_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (owns_dim) {
        acc = acc * shared_alpha + shared_beta * float(V[kv_base + int(tid)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (owns_dim) {
    O[q_base + int(tid)] = ELEM_TYPE(acc / max(shared_l, 1.0e-20f));
}
