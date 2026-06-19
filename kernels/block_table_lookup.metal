uint b = thread_position_in_grid.x;

const int B = meta[0];
const int MAX_BLOCKS = meta[1];
const int PAGE_SIZE = meta[2];
if (b >= uint(B)) {
    return;
}

const int pos = positions[b];
const int block_id = pos / PAGE_SIZE;
const int offset = pos % PAGE_SIZE;
page_ids[b] = block_table[b * MAX_BLOCKS + block_id];
offsets[b] = offset;
