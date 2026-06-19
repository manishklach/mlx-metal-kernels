uint elem = thread_position_in_grid.x;
const int total = meta[0];
if (elem >= uint(total)) {
    return;
}
y[elem] = ELEM_TYPE(float(x[elem]) + float(residual[elem]));
