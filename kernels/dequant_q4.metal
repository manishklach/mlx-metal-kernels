uint elem = thread_position_in_grid.x;

const int M = meta[0];
const int K_packed = meta[1];
const int K = meta[2];
const int group_size = meta[3];
const int has_zero = meta[4];
const uint total = uint(M * K);
if (elem >= total) {
    return;
}

const int k = int(elem % uint(K));
const int m = int(elem / uint(K));
const int byte_idx = m * K_packed + (k / 2);
const uchar byte_val = packed[byte_idx];
const float q = (k % 2 == 0) ? float(byte_val & 0x0F) : float((byte_val >> 4) & 0x0F);
const int group = k / group_size;
const float scale_val = scales[m * ((K + group_size - 1) / group_size) + group];
const float zero_val = has_zero ? zeros[m * ((K + group_size - 1) / group_size) + group] : 0.0f;
const float y = has_zero ? ((q - zero_val) * scale_val) : (q * scale_val);
out[elem] = ELEM_TYPE(y);
