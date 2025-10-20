from typing import Optional, Tuple

import numpy as np

import syna
from syna import utils
from syna.core import Function, Tensor, as_tensor


# ----------------------
# Shape manipulations
# ----------------------
class Reshape(Function):
    """Reshape tensor to a given shape."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape) -> Tensor:
    """Reshape tensor; if shape matches returns as_tensor(x)."""
    if x.shape == shape:
        return as_tensor(x)
    return Reshape(shape)(x)


class Transpose(Function):
    """Transpose with optional axes permutation."""

    def __init__(self, axes=None) -> None:
        self.axes = axes

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None) -> Tensor:
    """Transpose tensor along axes."""
    return Transpose(axes)(x)


class GetItem(Function):
    """Supports x[slices] and produces gradient via GetItemGrad."""

    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        return x[self.slices]

    def backward(self, gy):
        (x,) = self.inputs
        return GetItemGrad(self.slices, x.shape)(gy)


class GetItemGrad(Function):
    """Gradient for getitem: scatters gy back into the original shape."""

    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, x):
        gx = np.zeros(self.in_shape, dtype=x.dtype)
        np.add.at(gx, self.slices, x)
        return gx


def get_item(x, slices) -> Tensor:
    """Index into tensor with slices."""
    return GetItem(slices)(x)


def expand_dims(x, axis: int) -> Tensor:
    """Insert a dimension of size 1 at index axis."""
    x = as_tensor(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x) -> Tensor:
    """Flatten all dimensions except the first (batch) dimension."""
    return reshape(x, (x.shape[0], -1))


# ----------------------
# Reductions & broadcasting
# ----------------------
class Sum(Function):
    """Sum reduction, supports axis and keepdims."""

    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x, axis: Optional[Tuple[int, ...]] = None, keepdims=False) -> Tensor:
    """Sum elements along given axes."""
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    """Sum elements to target shape (inverse of broadcast_to)."""

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)


def sum_to(x, shape: Tuple[int, ...]) -> Tensor:
    """Sum elements of x so result has `shape`."""
    if x.shape == shape:
        return as_tensor(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    """Broadcast x to shape."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)


def broadcast_to(x, shape: Tuple[int, ...]) -> Tensor:
    """Broadcast x to the given shape."""
    if x.shape == shape:
        return as_tensor(x)
    return BroadcastTo(shape)(x)


def dropout(x, dropout_ratio=0.5) -> Tensor:
    """Dropout during training; identity during evaluation."""
    x = as_tensor(x)
    if syna.Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        return x * mask / scale
    return x
