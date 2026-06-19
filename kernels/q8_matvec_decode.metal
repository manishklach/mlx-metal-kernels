uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int K = meta[1];
const int N = meta[2];
const int group_size = meta[3];
const int has_zero = meta[4];
const uint total = uint(B * N);
if (row_id >= total) {
    return;
}

const int n = int(row_id % uint(N));
const int b = int(row_id / uint(N));
const int groups = (K + group_size - 1) / group_size;
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    const float qv = float(q_w[n * K + k]);
    const int group = k / group_size;
    const float scale_val = scales[n * groups + group];
    const float zero_val = has_zero ? zeros[n * groups + group] : 0.0f;
    const float w = has_zero ? ((qv - zero_val) * scale_val) : (qv * scale_val);
    acc += float(x[b * K + k]) * w;
}
y[b * N + n] = ELEM_TYPE(acc);
