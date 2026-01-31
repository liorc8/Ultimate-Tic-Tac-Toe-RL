import numpy as np
from ultimate_board import UltimateBoard

ACTION_SIZE = 81


def legal_action_mask(game: UltimateBoard) -> np.ndarray:
    """
    Create a binary mask over the action space.

    Shape: (81,)

    mask[a] == 1.0  -> action a is legal
    mask[a] == 0.0  -> action a is illegal
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)

    for action in game.legal_actions():
        mask[action] = 1.0

    return mask