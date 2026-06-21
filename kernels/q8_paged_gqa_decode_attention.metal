uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int NUM_PAGES = meta[1];
const int PAGE_SIZE = meta[2];
const int MAX_BLOCKS = meta[3];
const int Hq = meta[4];
const int Hkv = meta[5];
const int D = meta[6];
const int group_size = meta[7];
const int groups_per_head = meta[8];

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * Hq);
if (row_id >= total_rows) {
    return;
}

const int hq = int(row_id % uint(Hq));
const int b = int(row_id / uint(Hq));
const int group = Hq / Hkv;
const int hkv = hq / group;

const int q_stride = Hq * D;
const int page_stride_p = PAGE_SIZE * Hkv * D;
const int page_stride_s = Hkv * D;
const int page_stride_h = D;
const int scale_stride_p = PAGE_SIZE * Hkv * groups_per_head;
const int scale_stride_s = Hkv * groups_per_head;
const int scale_stride_h = groups_per_head;
const int bt_stride = MAX_BLOCKS;

const int valid_len = lengths[b];
if (valid_len <= 0) {
    return;
}

const int q_base = b * q_stride + hq * D;
const float scale_val = scale[0];

float row_m = -INFINITY;
float row_l = 0.0f;
float acc[MAX_HEAD_DIM];
for (int d = 0; d < MAX_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
}

for (int j = 0; j < valid_len; ++j) {
    const int block_idx = j / PAGE_SIZE;
    const int offset = j % PAGE_SIZE;
    const int page_id = block_table[b * bt_stride + block_idx];
    if (page_id < 0) {
        continue;
    }

    const int kv_base = page_id * page_stride_p + offset * page_stride_s + hkv * page_stride_h;
    const int sc_base = page_id * scale_stride_p + offset * scale_stride_s + hkv * scale_stride_h;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        const int g = d / group_size;
        const float q_val = float(q[q_base + d]);
        const float k_val = (float(K_pages_q[kv_base + d]) - 128.0f) * K_scales[sc_base + g];
        dot += q_val * k_val;
    }
    const float score = dot * scale_val;
    const float m_new = max(row_m, score);
    const float alpha = isfinite(row_m) ? exp(row_m - m_new) : 0.0f;
    const float beta = exp(score - m_new);
    row_l = row_l * alpha + beta;
    row_m = m_new;

    for (int d = 0; d < D; ++d) {
        const int g = d / group_size;
        const float v_val = (float(V_pages_q[kv_base + d]) - 128.0f) * V_scales[sc_base + g];
        acc[d] = acc[d] * alpha + beta * v_val;
    }
}

const float inv_l = 1.0f / max(row_l, 1.0e-20f);
for (int d = 0; d < D; ++d) {
    out[q_base + d] = ELEM_TYPE(acc[d] * inv_l);
}
