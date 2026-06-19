uint elem = thread_position_in_grid.x;

const int M = meta[0];
const int K = meta[1];
const int group_size = meta[2];
const int has_zero = meta[3];
const uint total = uint(M * K);
if (elem >= total) {
    return;
}

const int k = int(elem % uint(K));
const int m = int(elem / uint(K));
const int groups = (K + group_size - 1) / group_size;
const int group = k / group_size;
const float qv = float(q[elem]);
const float scale_val = scales[m * groups + group];
const float zero_val = has_zero ? zeros[m * groups + group] : 0.0f;
const float y = has_zero ? ((qv - zero_val) * scale_val) : (qv * scale_val);
out[elem] = ELEM_TYPE(y);
