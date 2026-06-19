uint pair_id = thread_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];
const int cos_rows = meta[4];
const int position_offset = meta[5];
const int input_layout = meta[6];
const int half_D = D / 2;
const uint total = uint(B * S * H * half_D);
if (pair_id >= total) {
    return;
}

const int pair = int(pair_id % uint(half_D));
const int h = int((pair_id / uint(half_D)) % uint(H));
const int s = int((pair_id / uint(half_D * H)) % uint(S));
const int b = int(pair_id / uint(half_D * H * S));
const int pos = position_offset + s;
if (pos < 0 || pos >= cos_rows) {
    return;
}

const int even = 2 * pair;
const int odd = even + 1;
const int out_base = ((b * S + s) * H + h) * D;

int q_even_idx;
int q_odd_idx;
int k_even_idx;
int k_odd_idx;
int v_even_idx;
int v_odd_idx;
if (input_layout == 0) {
    const int row_base = (b * S + s) * (3 * H * D);
    const int offset_even = h * D + even;
    const int offset_odd = h * D + odd;
    q_even_idx = row_base + offset_even;
    q_odd_idx = row_base + offset_odd;
    k_even_idx = row_base + H * D + offset_even;
    k_odd_idx = row_base + H * D + offset_odd;
    v_even_idx = row_base + 2 * H * D + offset_even;
    v_odd_idx = row_base + 2 * H * D + offset_odd;
} else {
    const int row_base = ((b * S + s) * 3 * H + h) * D;
    q_even_idx = row_base + even;
    q_odd_idx = row_base + odd;
    k_even_idx = row_base + H * D + even;
    k_odd_idx = row_base + H * D + odd;
    v_even_idx = row_base + 2 * H * D + even;
    v_odd_idx = row_base + 2 * H * D + odd;
}

const float c = cos[pos * half_D + pair];
const float sv = sin[pos * half_D + pair];
const float q_even = float(qkv[q_even_idx]);
const float q_odd = float(qkv[q_odd_idx]);
const float k_even = float(qkv[k_even_idx]);
const float k_odd = float(qkv[k_odd_idx]);

q_rope[out_base + even] = ELEM_TYPE(q_even * c - q_odd * sv);
q_rope[out_base + odd] = ELEM_TYPE(q_even * sv + q_odd * c);
k_rope[out_base + even] = ELEM_TYPE(k_even * c - k_odd * sv);
k_rope[out_base + odd] = ELEM_TYPE(k_even * sv + k_odd * c);
v[out_base + even] = qkv[v_even_idx];
v[out_base + odd] = qkv[v_odd_idx];
