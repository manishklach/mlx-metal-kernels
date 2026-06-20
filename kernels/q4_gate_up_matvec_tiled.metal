const uint tid = thread_position_in_threadgroup.x;
const uint tile_id = threadgroup_position_in_grid.x;
const uint tg_size = threads_per_threadgroup.x;

const int rows = meta[0];
const int K = meta[1];
const int N = meta[2];
const int K_packed = meta[3];
const int group_size = meta[4];
const int num_groups = meta[5];
const int has_gate_zero = meta[6];
const int has_up_zero = meta[7];
const int n_tile = meta[8];

const int tiles_per_row = (N + n_tile - 1) / n_tile;
const uint total_tiles = uint(rows * tiles_per_row);
if (tile_id >= total_tiles) {
    return;
}

const int row = int(tile_id / uint(tiles_per_row));
const int tile_idx = int(tile_id % uint(tiles_per_row));
const int n_base = tile_idx * n_tile;

threadgroup float gate_scratch0[MATVEC_TILED_THREADS];
threadgroup float gate_scratch1[MATVEC_TILED_THREADS];
threadgroup float gate_scratch2[MATVEC_TILED_THREADS];
threadgroup float gate_scratch3[MATVEC_TILED_THREADS];
threadgroup float up_scratch0[MATVEC_TILED_THREADS];
threadgroup float up_scratch1[MATVEC_TILED_THREADS];
threadgroup float up_scratch2[MATVEC_TILED_THREADS];
threadgroup float up_scratch3[MATVEC_TILED_THREADS];

float gate_partial0 = 0.0f;
float gate_partial1 = 0.0f;
float gate_partial2 = 0.0f;
float gate_partial3 = 0.0f;
float up_partial0 = 0.0f;
float up_partial1 = 0.0f;
float up_partial2 = 0.0f;
float up_partial3 = 0.0f;

for (int k = int(tid); k < K; k += int(tg_size)) {
    const float x_val = float(x[row * K + k]);
    const int packed_col = k / 2;
    const bool high_nibble = ((k & 1) != 0);
    const int g = min(k / group_size, num_groups - 1);

    for (int t = 0; t < 4; ++t) {
        const int n = n_base + t;
        if (t >= n_tile || n >= N) {
            continue;
        }

        const uint8_t gate_p = gate_packed[n * K_packed + packed_col];
        const float gate_q = high_nibble ? float((gate_p >> 4) & 0x0F) : float(gate_p & 0x0F);
        const float gate_scale = float(gate_scales[n * num_groups + g]);
        const float gate_zero = has_gate_zero ? float(gate_zeros[n * num_groups + g]) : 0.0f;
        const float gate_w_val = has_gate_zero ? (gate_q - gate_zero) * gate_scale : gate_q * gate_scale;

        const uint8_t up_p = up_packed[n * K_packed + packed_col];
        const float up_q = high_nibble ? float((up_p >> 4) & 0x0F) : float(up_p & 0x0F);
        const float up_scale = float(up_scales[n * num_groups + g]);
        const float up_zero = has_up_zero ? float(up_zeros[n * num_groups + g]) : 0.0f;
        const float up_w_val = has_up_zero ? (up_q - up_zero) * up_scale : up_q * up_scale;

        if (t == 0) {
            gate_partial0 += x_val * gate_w_val;
            up_partial0 += x_val * up_w_val;
        } else if (t == 1) {
            gate_partial1 += x_val * gate_w_val;
            up_partial1 += x_val * up_w_val;
        } else if (t == 2) {
            gate_partial2 += x_val * gate_w_val;
            up_partial2 += x_val * up_w_val;
        } else {
            gate_partial3 += x_val * gate_w_val;
            up_partial3 += x_val * up_w_val;
        }
    }
}

gate_scratch0[tid] = gate_partial0;
gate_scratch1[tid] = gate_partial1;
gate_scratch2[tid] = gate_partial2;
gate_scratch3[tid] = gate_partial3;
up_scratch0[tid] = up_partial0;
up_scratch1[tid] = up_partial1;
up_scratch2[tid] = up_partial2;
up_scratch3[tid] = up_partial3;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint offset = tg_size / 2u; offset > 0u; offset >>= 1u) {
    if (tid < offset) {
        gate_scratch0[tid] += gate_scratch0[tid + offset];
        gate_scratch1[tid] += gate_scratch1[tid + offset];
        gate_scratch2[tid] += gate_scratch2[tid + offset];
        gate_scratch3[tid] += gate_scratch3[tid + offset];
        up_scratch0[tid] += up_scratch0[tid + offset];
        up_scratch1[tid] += up_scratch1[tid + offset];
        up_scratch2[tid] += up_scratch2[tid + offset];
        up_scratch3[tid] += up_scratch3[tid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    if (n_base + 0 < N && n_tile > 0) {
        gate[row * N + (n_base + 0)] = ELEM_TYPE(gate_scratch0[0]);
        up[row * N + (n_base + 0)] = ELEM_TYPE(up_scratch0[0]);
    }
    if (n_base + 1 < N && n_tile > 1) {
        gate[row * N + (n_base + 1)] = ELEM_TYPE(gate_scratch1[0]);
        up[row * N + (n_base + 1)] = ELEM_TYPE(up_scratch1[0]);
    }
    if (n_base + 2 < N && n_tile > 2) {
        gate[row * N + (n_base + 2)] = ELEM_TYPE(gate_scratch2[0]);
        up[row * N + (n_base + 2)] = ELEM_TYPE(up_scratch2[0]);
    }
    if (n_base + 3 < N && n_tile > 3) {
        gate[row * N + (n_base + 3)] = ELEM_TYPE(gate_scratch3[0]);
        up[row * N + (n_base + 3)] = ELEM_TYPE(up_scratch3[0]);
    }
}
