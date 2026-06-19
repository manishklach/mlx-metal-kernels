const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simd_id = simdgroup_index_in_threadgroup;
const uint tg_size = threads_per_threadgroup.x;
const uint num_simdgroups = (tg_size + 31u) / 32u;
const uint row_id = threadgroup_position_in_grid.x;

const int rows = meta[0];
const int D = meta[1];
if (row_id >= uint(rows)) {
    return;
}

const int row_base = int(row_id) * D;
threadgroup float simd_partials[32];
threadgroup float inv_rms_shared;

float partial = 0.0f;
for (uint d = tid; d < uint(D); d += tg_size) {
    const float zv = float(x[row_base + int(d)]) + float(residual[row_base + int(d)]);
    partial += zv * zv;
}

float reduced = simd_sum(partial);
if (lane == 0) {
    simd_partials[simd_id] = reduced;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < 32u) {
    float block_sum = (tid < num_simdgroups) ? simd_partials[tid] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (tid == 0) {
        inv_rms_shared = rsqrt(block_sum / float(D) + eps[0]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint d = tid; d < uint(D); d += tg_size) {
    const float zv = float(x[row_base + int(d)]) + float(residual[row_base + int(d)]);
    z[row_base + int(d)] = ELEM_TYPE(zv);
    y[row_base + int(d)] = ELEM_TYPE(zv * inv_rms_shared * float(weight[int(d)]));
}
