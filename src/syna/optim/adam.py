import math
from typing import Dict

import numpy as np

from syna.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer with bias-correction handled via dynamic lr property.

    alpha: base step size
    beta1, beta2: exponential decay rates for first and second moment estimates
    """

    def __init__(
        self,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._ms: Dict[int, np.ndarray] = {}
        self._vs: Dict[int, np.ndarray] = {}

    def update(self, *args, **kwargs) -> None:
        """Increment time step and perform parameter updates."""
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self) -> float:
        """Compute bias-corrected learning rate factor."""
        fix1 = 1.0 - math.pow(self.beta1, self.t)
        fix2 = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param) -> None:
        m = self._state(self._ms, param)
        v = self._state(self._vs, param)
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (np.sqrt(v) + eps)
