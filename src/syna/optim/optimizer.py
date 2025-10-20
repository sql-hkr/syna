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
