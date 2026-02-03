import numpy as np
import torch

from game_network import GameNetwork


class TrainedNetwork:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = GameNetwork.load(model_path, map_location=device)
        self.model.to(torch.device(device))
        self.model.eval()

    def predict(self, state: np.ndarray):
        """
        Args:
            state: np.ndarray (C,9,9) float32

        Returns:
            policy: np.ndarray (81,) float32, sums ~1
            value: float in [-1,1]
        """
        policy, value = self.model.predict(state, device=self.device)
        return policy, value
