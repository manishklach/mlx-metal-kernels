uint pair_id = thread_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];
const int cos_rows = meta[4];
const int position_offset = meta[5];

const int half_D = D / 2;
const uint total_pairs = uint(B * S * H * half_D);
if (pair_id >= total_pairs) {
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

const int base = ((b * S + s) * H + h) * D;
const int even_idx = base + 2 * pair;
const int odd_idx = even_idx + 1;
const int cs_idx = pos * half_D + pair;

const float x_even = float(x[even_idx]);
const float x_odd = float(x[odd_idx]);
const float c = cos[cs_idx];
const float s_val = sin[cs_idx];

y[even_idx] = ELEM_TYPE(x_even * c - x_odd * s_val);
y[odd_idx] = ELEM_TYPE(x_even * s_val + x_odd * c);
