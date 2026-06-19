uint elem = thread_position_in_grid.x;

const int NUM_PAGES = meta[0];
const int PAGE_SIZE = meta[1];
const int H = meta[2];
const int D = meta[3];
const int B = meta[4];
const int MAX_BLOCKS = meta[5];
const uint total = uint(NUM_PAGES * PAGE_SIZE * H * D);
if (elem >= total) {
    return;
}

const int d = int(elem % uint(D));
const int h = int((elem / uint(D)) % uint(H));
const int offset = int((elem / uint(D * H)) % uint(PAGE_SIZE));
const int page = int(elem / uint(D * H * PAGE_SIZE));
const int idx = ((page * PAGE_SIZE + offset) * H + h) * D + d;

updated_K_pages[idx] = K_pages[idx];
updated_V_pages[idx] = V_pages[idx];

for (int b = 0; b < B; ++b) {
    const int pos = positions[b];
    const int block_id = pos / PAGE_SIZE;
    const int target_offset = pos % PAGE_SIZE;
    const int page_id = block_table[b * MAX_BLOCKS + block_id];
    if (page == page_id && offset == target_offset) {
        const int token_idx = (b * H + h) * D + d;
        updated_K_pages[idx] = k_new[token_idx];
        updated_V_pages[idx] = v_new[token_idx];
    }
}
