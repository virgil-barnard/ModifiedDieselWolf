from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    """Wrap another optimizer implementing the Lookahead mechanism."""

    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        if not 1 <= k:
            raise ValueError("k must be at least 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups
        # Create slow weights
        self.slow_weights: List[torch.Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                param = p.detach().clone()
                param.requires_grad = False
                self.slow_weights.append(param)

    def zero_grad(self) -> None:  # type: ignore[override]
        self.optimizer.zero_grad()

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:  # type: ignore[override]
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        idx += 1
                        continue
                    slow = self.slow_weights[idx]
                    slow.add_(self.alpha * (p.data - slow))
                    p.data.copy_(slow)
                    idx += 1
        return loss

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "slow_weights": [w.clone() for w in self.slow_weights],
            "step_counter": self.step_counter,
            "k": self.k,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        for sw, saved in zip(self.slow_weights, state_dict["slow_weights"]):
            sw.data.copy_(saved)
        self.step_counter = state_dict["step_counter"]
        self.k = state_dict["k"]
        self.alpha = state_dict["alpha"]

    @property
    def state(self) -> Dict:
        return self.optimizer.state

    @property
    def defaults(self) -> Dict:
        """Return defaults of the wrapped optimizer."""
        return getattr(self.optimizer, "defaults", {})
