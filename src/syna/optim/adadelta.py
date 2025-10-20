from typing import Dict

import numpy as np

from syna.optim.optimizer import Optimizer


class AdaDelta(Optimizer):
    """AdaDelta optimizer without a global learning rate."""

    def __init__(self, rho: float = 0.95, eps: float = 1e-6) -> None:
        super().__init__()
        self.rho = rho
        self.eps = eps
        self._msg: Dict[int, np.ndarray] = {}
        self._msdx: Dict[int, np.ndarray] = {}

    def update_one(self, param) -> None:
        msg = self._state(self._msg, param)
        msdx = self._state(self._msdx, param)
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = np.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx
