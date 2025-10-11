"""
Optimization algorithms and helper utilities.

Includes common optimizers (SGD, Adam, AdaGrad, etc.) and small utilities
that are used as hooks (weight decay, gradient clipping, parameter freezing).
"""

import math
from typing import Callable, Dict, Iterable, List

import numpy as np

import syna


class Optimizer:
    """Base optimizer.

    Subclasses should implement update_one(param). The optimizer keeps a
    reference to the target (model) via setup(target) and supports hooks
    that run before updates.
    """

    def __init__(self) -> None:
        self.target = None
        self.hooks: List[Callable[[Iterable], None]] = []

    def setup(self, target):
        """Attach optimizer to a target (model) which must provide params()."""
        self.target = target
        return self

    def update(self) -> None:
        """Run hooks and update all parameters with non-None gradients."""
        params = [p for p in self.target.params() if p.grad is not None]
        for f in self.hooks:
            f(params)
        for param in params:
            self.update_one(param)

    def update_one(self, param) -> None:
        """Update a single parameter. Must be implemented by subclasses."""
        raise NotImplementedError()

    def add_hook(self, f: Callable[[Iterable], None]) -> None:
        """Add a hook function called with the list of parameters before updates."""
        self.hooks.append(f)

    # Utility for managing per-parameter state dicts (e.g., moments, accumulators).
    def _state(self, store: Dict[int, np.ndarray], param: object) -> np.ndarray:
        key = id(param)
        if key not in store:
            store[key] = np.zeros_like(param.data)
        return store[key]


class WeightDecay:
    """L2 weight decay applied to gradients.

    rate: multiplier applied to parameter data and added to gradient.
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, params: Iterable) -> None:
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad:
    """Clip gradients by global norm.

    max_norm: maximum allowed norm for concatenated gradients.
    """

    def __init__(self, max_norm: float) -> None:
        self.max_norm = max_norm

    def __call__(self, params: Iterable) -> None:
        total = 0.0
        for param in params:
            g = param.grad.data
            total += float((g * g).sum())
        total_norm = math.sqrt(total)
        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1.0:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    """Freeze specified parameters or layers (set their grads to None)."""

    def __init__(self, *layers) -> None:
        self.freeze_params = []
        for layer in layers:
            if isinstance(layer, syna.Parameter):
                self.freeze_params.append(layer)
            else:
                for p in layer.params():
                    self.freeze_params.append(p)

    def __call__(self, params: Iterable) -> None:
        for p in self.freeze_params:
            p.grad = None


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param) -> None:
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    """SGD with classical momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self._vs: Dict[int, np.ndarray] = {}

    def update_one(self, param) -> None:
        v = self._state(self._vs, param)
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


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
