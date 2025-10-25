"""Normalization layers"""

from __future__ import annotations

import numpy as np

import syna.functions as F
from syna.core import Parameter
from syna.layers.layer import Layer


class LayerNorm(Layer):
    r"""Layer Normalization (Jimmy Lei Ba et al. 2016).

    paper: https://arxiv.org/abs/1607.06450

    .. math::
        y = \gamma \odot \frac{x - \mathbb{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} + \beta

    Args:
        dim: size of the last (normalized) dimension
        eps: small constant added to variance for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-5, dtype=np.float32) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=dtype), name="gamma")
        self.beta = Parameter(np.zeros(dim, dtype=dtype), name="beta")

    def forward(self, x):
        """Apply layer normalization to input tensor."""
        axis = len(x.shape) - 1
        mean = F.mean(x, axis=axis, keepdims=True)
        diff = x - mean
        var = F.mean(diff * diff, axis=axis, keepdims=True)
        std = F.sqrt(var + self.eps)
        x_hat = diff / std
        return x_hat * self.gamma + self.beta
