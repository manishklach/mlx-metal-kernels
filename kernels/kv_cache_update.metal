uint elem = thread_position_in_grid.x;

const int B = meta[0];
const int MAX_S = meta[1];
const int H = meta[2];
const int D = meta[3];

const uint total_elems = uint(B * MAX_S * H * D);
if (elem >= total_elems) {
    return;
}

const int d = int(elem % uint(D));
const int h = int((elem / uint(D)) % uint(H));
const int s = int((elem / uint(D * H)) % uint(MAX_S));
const int b = int(elem / uint(D * H * MAX_S));

const int cache_idx = ((b * MAX_S + s) * H + h) * D + d;
const int token_idx = ((b * 1 + 0) * H + h) * D + d;
const int pos = positions[b];

if (s == pos) {
    updated_K_cache[cache_idx] = k_new[token_idx];
    updated_V_cache[cache_idx] = v_new[token_idx];
} else {
    updated_K_cache[cache_idx] = K_cache[cache_idx];
    updated_V_cache[cache_idx] = V_cache[cache_idx];
}
