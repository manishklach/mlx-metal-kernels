uint elem = thread_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];
const int input_layout = meta[4];
const uint total = uint(B * S * H * D);
if (elem >= total) {
    return;
}

const int d = int(elem % uint(D));
const int h = int((elem / uint(D)) % uint(H));
const int s = int((elem / uint(D * H)) % uint(S));
const int b = int(elem / uint(D * H * S));
const int out_idx = ((b * S + s) * H + h) * D + d;

int q_idx;
int k_idx;
int v_idx;
if (input_layout == 0) {
    const int row_base = (b * S + s) * (3 * H * D);
    const int offset = h * D + d;
    q_idx = row_base + offset;
    k_idx = row_base + H * D + offset;
    v_idx = row_base + 2 * H * D + offset;
} else {
    const int row_base = ((b * S + s) * 3 * H + h) * D + d;
    q_idx = row_base;
    k_idx = row_base + H * D;
    v_idx = row_base + 2 * H * D;
}

q[out_idx] = qkv[q_idx];
k[out_idx] = qkv[k_idx];
v[out_idx] = qkv[v_idx];
