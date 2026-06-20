const uint tid = thread_position_in_threadgroup.x;
const uint tile_id = threadgroup_position_in_grid.x;
const uint tg_size = threads_per_threadgroup.x;

const int B = meta[0];
const int N = meta[1];
const int K = meta[2];
const int K_packed = meta[3];
const int group_size = meta[4];
const int num_groups = meta[5];
const int has_zero = meta[6];
const int n_tile = meta[7];

const int tiles_per_batch = (N + n_tile - 1) / n_tile;
const uint total_tiles = uint(B * tiles_per_batch);
if (tile_id >= total_tiles) {
    return;
}

const int b = int(tile_id / uint(tiles_per_batch));
const int tile_idx = int(tile_id % uint(tiles_per_batch));
const int n_base = tile_idx * n_tile;

threadgroup float scratch0[MATVEC_TILED_THREADS];
threadgroup float scratch1[MATVEC_TILED_THREADS];
threadgroup float scratch2[MATVEC_TILED_THREADS];
threadgroup float scratch3[MATVEC_TILED_THREADS];

float partial0 = 0.0f;
float partial1 = 0.0f;
float partial2 = 0.0f;
float partial3 = 0.0f;

for (int k = int(tid); k < K; k += int(tg_size)) {
    const float x_val = float(x[b * K + k]);
    const int packed_col = k / 2;
    const bool high_nibble = ((k & 1) != 0);

    for (int t = 0; t < 4; ++t) {
        const int n = n_base + t;
        if (t >= n_tile || n >= N) {
            continue;
        }
        const uint8_t packed = packed_w[n * K_packed + packed_col];
        const float q_val = high_nibble ? float((packed >> 4) & 0x0F) : float(packed & 0x0F);
        const int g = min(k / group_size, num_groups - 1);
        const float scale_val = float(scales[n * num_groups + g]);
        const float zero_val = has_zero ? float(zeros[n * num_groups + g]) : 0.0f;
        const float w_val = has_zero ? (q_val - zero_val) * scale_val : q_val * scale_val;
        if (t == 0) {
            partial0 += x_val * w_val;
        } else if (t == 1) {
            partial1 += x_val * w_val;
        } else if (t == 2) {
            partial2 += x_val * w_val;
        } else {
            partial3 += x_val * w_val;
        }
    }
}

scratch0[tid] = partial0;
scratch1[tid] = partial1;
scratch2[tid] = partial2;
scratch3[tid] = partial3;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint offset = tg_size / 2u; offset > 0u; offset >>= 1u) {
    if (tid < offset) {
        scratch0[tid] += scratch0[tid + offset];
        scratch1[tid] += scratch1[tid + offset];
        scratch2[tid] += scratch2[tid + offset];
        scratch3[tid] += scratch3[tid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    if (n_base + 0 < N && n_tile > 0) {
        y[b * N + (n_base + 0)] = ELEM_TYPE(scratch0[0]);
    }
    if (n_base + 1 < N && n_tile > 1) {
        y[b * N + (n_base + 1)] = ELEM_TYPE(scratch1[0]);
    }
    if (n_base + 2 < N && n_tile > 2) {
        y[b * N + (n_base + 2)] = ELEM_TYPE(scratch2[0]);
    }
    if (n_base + 3 < N && n_tile > 3) {
        y[b * N + (n_base + 3)] = ELEM_TYPE(scratch3[0]);
    }
}
