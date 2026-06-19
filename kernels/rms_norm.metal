const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simd_id = simdgroup_index_in_threadgroup;
const uint tg_size = threads_per_threadgroup.x;
const uint num_simdgroups = (tg_size + 31u) / 32u;
const uint row_id = threadgroup_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int D = meta[2];

const uint total_rows = uint(B * S);
if (row_id >= total_rows) {
    return;
}

const int b = int(row_id / uint(S));
const int s = int(row_id % uint(S));
const int row_base = (b * S + s) * D;

threadgroup float simd_partials[32];
threadgroup float inv_rms_shared;

float partial_sum = 0.0f;
for (uint d = tid; d < uint(D); d += tg_size) {
    const float val = float(x[row_base + int(d)]);
    partial_sum += val * val;
}

float reduced = simd_sum(partial_sum);
if (lane == 0) {
    simd_partials[simd_id] = reduced;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < 32u) {
    float block_sum = (tid < num_simdgroups) ? simd_partials[tid] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (tid == 0) {
        const float variance = block_sum / float(D);
        inv_rms_shared = rsqrt(variance + eps[0]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint d = tid; d < uint(D); d += tg_size) {
    const float val = float(x[row_base + int(d)]);
    const float w = float(weight[int(d)]);
    y[row_base + int(d)] = ELEM_TYPE(val * inv_rms_shared * w);
}
