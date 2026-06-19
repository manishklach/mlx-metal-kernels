uint elem = thread_position_in_grid.x;

const int B = meta[0];
const int MAX_S = meta[1];
const int H = meta[2];
const int D = meta[3];
const int cos_rows = meta[4];
const int input_layout = meta[5];
const uint total = uint(B * MAX_S * H * D);
if (elem >= total) {
    return;
}

const int d = int(elem % uint(D));
const int h = int((elem / uint(D)) % uint(H));
const int s = int((elem / uint(D * H)) % uint(MAX_S));
const int b = int(elem / uint(D * H * MAX_S));
const int pos = positions[b];
const int cache_idx = ((b * MAX_S + s) * H + h) * D + d;

if (s != pos) {
    updated_K_cache[cache_idx] = K_cache[cache_idx];
    updated_V_cache[cache_idx] = V_cache[cache_idx];
    return;
}

const int pair = d / 2;
const bool is_even = (d % 2) == 0;
const int even = is_even ? d : d - 1;
const int odd = is_even ? d + 1 : d;
if (pos < 0 || pos >= cos_rows) {
    updated_K_cache[cache_idx] = K_cache[cache_idx];
    updated_V_cache[cache_idx] = V_cache[cache_idx];
    return;
}

int q_even_idx;
int q_odd_idx;
int k_even_idx;
int k_odd_idx;
int v_idx;
if (input_layout == 0) {
    const int row_base = b * (3 * H * D);
    q_even_idx = row_base + h * D + even;
    q_odd_idx = row_base + h * D + odd;
    k_even_idx = row_base + H * D + h * D + even;
    k_odd_idx = row_base + H * D + h * D + odd;
    v_idx = row_base + 2 * H * D + h * D + d;
} else {
    const int row_base = (b * 3 * H + h) * D;
    q_even_idx = row_base + even;
    q_odd_idx = row_base + odd;
    k_even_idx = row_base + H * D + even;
    k_odd_idx = row_base + H * D + odd;
    v_idx = row_base + 2 * H * D + d;
}

const float c = cos[pos * (D / 2) + pair];
const float sv = sin[pos * (D / 2) + pair];
const float q_even = float(qkv[q_even_idx]);
const float q_odd = float(qkv[q_odd_idx]);
const float k_even = float(qkv[k_even_idx]);
const float k_odd = float(qkv[k_odd_idx]);
const float q_val = is_even ? (q_even * c - q_odd * sv) : (q_even * sv + q_odd * c);
const float k_val = is_even ? (k_even * c - k_odd * sv) : (k_even * sv + k_odd * c);

const int q_out_idx = (b * H + h) * D + d;
q_rope[q_out_idx] = ELEM_TYPE(q_val);
updated_K_cache[cache_idx] = ELEM_TYPE(k_val);
updated_V_cache[cache_idx] = qkv[v_idx];
