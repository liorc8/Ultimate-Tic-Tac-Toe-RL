def encode_action(br, bc, sr, sc):
    """
    Encode a move (br, bc, sr, sc) into a single integer action in [0, 80].
    """
    if not (0 <= br < 3 and 0 <= bc < 3 and 0 <= sr < 3 and 0 <= sc < 3):
        raise ValueError("Indices out of bounds")

    small_index = 3 * br + bc      # 0..8
    cell_index = 3 * sr + sc       # 0..8
    return 9 * small_index + cell_index


def decode_action(action):
    """
    Decode an integer action in [0, 80] into (br, bc, sr, sc).
    """
    if not (0 <= action < 81):
        raise ValueError("Action out of bounds")

    small_index = action // 9      # 0..8
    cell_index = action % 9        # 0..8

    br, bc = divmod(small_index, 3)
    sr, sc = divmod(cell_index, 3)
    return br, bc, sr, sc
