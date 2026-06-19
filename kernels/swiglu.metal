uint elem = thread_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int D = meta[2];
const uint total_elems = uint(B * S * D);
if (elem >= total_elems) {
    return;
}

const float gate_val = float(gate[elem]);
const float up_val = float(up[elem]);
const float silu = gate_val / (1.0f + exp(-gate_val));
out[elem] = ELEM_TYPE(silu * up_val);
