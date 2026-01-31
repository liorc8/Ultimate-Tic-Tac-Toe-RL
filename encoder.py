import numpy as np
from ultimate_board import UltimateBoard
from small_board import SmallBoard


# ===== Encoder configuration =====
STATE_CHANNELS = 4
BOARD_SIZE = 9


def encode_state(game: UltimateBoard) -> np.ndarray:
    """
    Encode the UltimateBoard state into a tensor for a neural network.

    Shape: (4, 9, 9)

    Channel 0: Current player's stones
    Channel 1: Opponent's stones
    Channel 2: Active / allowed board regions
    Channel 3: Closed small boards (won or draw)
    """
    # Initialize empty state tensor
    state = np.zeros(
        (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32
    )

    current = game.current_player
    opponent = (
        SmallBoard.O if current == SmallBoard.X else SmallBoard.X
    )

    # ---- Channel 0 & 1: stones ----
    for br in range(3):
        for bc in range(3):
            sb = game.boards[br][bc]

            for sr in range(3):
                for sc in range(3):
                    global_row = br * 3 + sr
                    global_col = bc * 3 + sc

                    cell = sb.board[sr][sc]

                    if cell == current:
                        state[0, global_row, global_col] = 1.0
                    elif cell == opponent:
                        state[1, global_row, global_col] = 1.0

    # ---- Channel 2: active board mask ----
    if game.active_board is not None:
        abr, abc = game.active_board
        for sr in range(3):
            for sc in range(3):
                global_row = abr * 3 + sr
                global_col = abc * 3 + sc
                state[2, global_row, global_col] = 1.0
    else:
        # All open boards are allowed
        for br in range(3):
            for bc in range(3):
                if game.boards[br][bc].is_open():
                    for sr in range(3):
                        for sc in range(3):
                            global_row = br * 3 + sr
                            global_col = bc * 3 + sc
                            state[2, global_row, global_col] = 1.0

    # ---- Channel 3: closed small boards ----
    for br in range(3):
        for bc in range(3):
            sb = game.boards[br][bc]
            if not sb.is_open():
                for sr in range(3):
                    for sc in range(3):
                        global_row = br * 3 + sr
                        global_col = bc * 3 + sc
                        state[3, global_row, global_col] = 1.0

    return state

