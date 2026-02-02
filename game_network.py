import numpy as np


class GameNetwork:
    """
    Dummy neural network for Ultimate Tic-Tac-Toe.

    This implementation is intentionally simple:
    - Uniform policy over all actions
    - Zero value for all states

    It allows the rest of the system (PUCT/MCTS) to be developed
    without depending on a trained neural network.
    """

    ACTION_SIZE = 81

    def __init__(self):
        pass

    def predict(self, state: np.ndarray):
        """
        Predict policy and value for a given game state.

        Input:
            state: np.ndarray of shape (4, 9, 9)

        Output:
            policy: np.ndarray of shape (81,), sums to 1.0
            value: float (always 0.0 in this dummy version)
        """
        if state.shape != (4, 9, 9):
            raise ValueError(f"Expected state shape (4, 9, 9), got {state.shape}")

        policy = np.ones(self.ACTION_SIZE, dtype=np.float32)
        policy /= policy.sum()  # uniform distribution

        value = 0.0

        return policy, value

    def save(self, path: str):
        """
        Dummy save (no parameters to save yet).
        """
        pass

    def load(self, path: str):
        """
        Dummy load (no parameters to load yet).
        """
        pass
