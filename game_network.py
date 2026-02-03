import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GameNetworkConfig:
    in_channels: int = 4
    board_h: int = 9
    board_w: int = 9
    hidden: int = 256
    policy_size: int = 81


class GameNetwork(nn.Module):
    def __init__(self, cfg: Optional[GameNetworkConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else GameNetworkConfig()

        in_features = self.cfg.in_channels * self.cfg.board_h * self.cfg.board_w

        # Shared trunk (simple MLP - good for debugging pre-training)
        self.fc1 = nn.Linear(in_features, self.cfg.hidden)
        self.fc2 = nn.Linear(self.cfg.hidden, self.cfg.hidden)

        # Policy head (logits for 81 actions)
        self.policy_head = nn.Linear(self.cfg.hidden, self.cfg.policy_size)

        # Value head (scalar in [-1, 1])
        self.value_head1 = nn.Linear(self.cfg.hidden, 64)
        self.value_head2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, C, 9, 9)

        Returns:
            policy_logits: (B, 81)
            value: (B,) in [-1, 1]
        """
        b = x.size(0)
        x = x.view(b, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)

        v = F.relu(self.value_head1(x))
        v = torch.tanh(self.value_head2(v)).squeeze(1)

        return policy_logits, v

    @torch.no_grad()
    def predict(self, state: np.ndarray, device: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Predict (policy, value) for a single state.

        Args:
            state: numpy array (C, 9, 9) float32
            device: "cpu" or "cuda" (optional)

        Returns:
            policy: numpy array (81,) normalized probabilities
            value: float in [-1, 1]
        """
        if state.ndim != 3:
            raise ValueError("state must have shape (C,9,9)")

        dev = torch.device(device) if device is not None else next(self.parameters()).device

        x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(dev)  # (1,C,9,9)
        policy_logits, value = self.forward(x)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        v = float(value.item())
        return policy, v

    def save(self, path: str):
        """
        Save model weights + config.
        """
        payload = {
            "state_dict": self.state_dict(),
            "config": self.cfg.__dict__,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "GameNetwork":
        """
        Load model weights + config.
        """
        payload = torch.load(path, map_location=map_location)
        cfg = GameNetworkConfig(**payload["config"])
        model = GameNetwork(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
