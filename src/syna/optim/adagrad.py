from typing import Dict

import numpy as np

from syna.optim.optimizer import Optimizer


class AdaGrad(Optimizer):
    """AdaGrad adaptive learning rate."""

    def __init__(self, lr: float = 0.001, eps: float = 1e-8) -> None:
        super().__init__()
        self.lr = lr
        self.eps = eps
        self._hs: Dict[int, np.ndarray] = {}

    def update_one(self, param) -> None:
        h = self._state(self._hs, param)
        grad = param.grad.data
        h += grad * grad
        param.data -= self.lr * grad / (np.sqrt(h) + self.eps)
