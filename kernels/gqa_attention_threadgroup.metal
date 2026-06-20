const uint tid = thread_position_in_threadgroup.x;
const uint row_id = threadgroup_position_in_grid.x;
const uint tg_size = threads_per_threadgroup.x;

const int B = meta[0];
const int Sq = meta[1];
const int Sk = meta[2];
const int Hq = meta[3];
const int Hkv = meta[4];
const int D = meta[5];
const bool causal = (meta[6] != 0);

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * Sq * Hq);
if (row_id >= total_rows) {
    return;
}

const int hq = int(row_id % uint(Hq));
const int q_idx = int((row_id / uint(Hq)) % uint(Sq));
const int b = int(row_id / uint(Sq * Hq));
const int group = Hq / Hkv;
const int hkv = hq / group;

const int q_stride_b = Sq * Hq * D;
const int q_stride_s = Hq * D;
const int q_stride_h = D;
const int kv_stride_b = Sk * Hkv * D;
const int kv_stride_s = Hkv * D;
const int kv_stride_h = D;

const int q_base = b * q_stride_b + q_idx * q_stride_s + hq * q_stride_h;
const int j_max = causal ? (q_idx + 1) : Sk;
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

for (int k_idx = 0; k_idx < j_max; ++k_idx) {
    const int kv_base = b * kv_stride_b + k_idx * kv_stride_s + hkv * kv_stride_h;

    float partial_dot = 0.0f;
    for (uint d = tid; d < uint(D); d += tg_size) {
        partial_dot += q_shared[d] * float(K[kv_base + int(d)]);
    }
    scratch[tid] = partial_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = tg_size / 2u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
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
