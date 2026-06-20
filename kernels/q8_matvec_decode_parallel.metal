uint gid = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint row_id = gid / MATVEC_THREADS;

const int B = meta[0];
const int N = meta[1];
const int K = meta[2];
const int group_size = meta[3];
const int num_groups = meta[4];
const int has_zero = meta[5];
const uint total = uint(B * N);
if (row_id >= total || tid >= MATVEC_THREADS) {
    return;
}

const int n = int(row_id % uint(N));
const int b = int(row_id / uint(N));
threadgroup float partial[MATVEC_THREADS];

float acc = 0.0f;
for (int k = int(tid); k < K; k += MATVEC_THREADS) {
    const float qv = float(q_w[n * K + k]);
    const int group = min(k / group_size, num_groups - 1);
    const float scale_val = scales[n * num_groups + group];
    const float zero_val = has_zero ? zeros[n * num_groups + group] : 0.0f;
    const float w = has_zero ? ((qv - zero_val) * scale_val) : (qv * scale_val);
    acc += float(x[b * K + k]) * w;
}

partial[tid] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = MATVEC_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    y[b * N + n] = ELEM_TYPE(partial[0]);
}
