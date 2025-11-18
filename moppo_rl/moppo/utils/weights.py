"""Utilities for preference-weight sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class WeightSampler:
    reward_dim: int
    strategy: Literal["dirichlet", "uniform"] = "dirichlet"
    dirichlet_alpha: float = 1.0

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.strategy == "dirichlet":
            alpha = np.full(self.reward_dim, self.dirichlet_alpha, dtype=np.float32)
            weights = np.random.dirichlet(alpha, size=batch_size).astype(np.float32)
        elif self.strategy == "uniform":
            weights = np.random.rand(batch_size, self.reward_dim).astype(np.float32)
            weights /= np.clip(weights.sum(axis=-1, keepdims=True), 1e-6, None)
        else:
            raise ValueError(f"Unsupported weight sampling strategy: {self.strategy}")
        return torch.from_numpy(weights)


__all__ = ["WeightSampler"]
