from typing import Optional, Tuple

import numpy as np

from syna import utils
from syna.core import Function, Tensor, as_tensor
from syna.functions.function import broadcast_to, reshape, sum, sum_to


class Add(Function):
    """Elementwise add with broadcasting support."""

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy: np.ndarray):
        if self.x0_shape != self.x1_shape:
            return sum_to(gy, self.x0_shape), sum_to(gy, self.x1_shape)
        return gy, gy


def add(x0, x1) -> Tensor:
    """Add two tensors (supports broadcasting)."""
    return Add()(x0, x1)


class Mul(Function):
    """Elementwise multiply with broadcasting support."""

    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            return sum_to(gx0, x0.shape), sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1) -> Tensor:
    """Elementwise multiply two tensors (supports broadcasting)."""
    return Mul()(x0, x1)


class Neg(Function):
    """Negation (unary -)."""

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x) -> Tensor:
    """Return -x."""
    return Neg()(x)


class Sub(Function):
    """Elementwise subtraction with broadcasting support."""

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        if self.x0_shape != self.x1_shape:
            return sum_to(gy, self.x0_shape), sum_to(-gy, self.x1_shape)
        return gy, -gy


def sub(x0, x1) -> Tensor:
    """Subtract x1 from x0 (supports broadcasting)."""
    return Sub()(x0, x1)


class Div(Function):
    """Elementwise division with broadcasting support."""

    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:
            return sum_to(gx0, x0.shape), sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1) -> Tensor:
    """Divide x0 by x1 (supports broadcasting)."""
    return Div()(x0, x1)


class Pow(Function):
    """Power operation x**c where c is a constant."""

    def __init__(self, c: float) -> None:
        self.c = c

    def forward(self, x):
        return x**self.c

    def backward(self, gy):
        x = self.inputs[0]
        return self.c * x ** (self.c - 1) * gy


def pow(x, c) -> Tensor:
    """Raise x to the constant power c."""
    return Pow(c)(x)


# ----------------------
# Trigonometric & exp/log
# ----------------------
class Sin(Function):
    """sin(x)"""

    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy * cos(self.inputs[0])


def sin(x) -> Tensor:
    """Sine of x."""
    return Sin()(x)


class Cos(Function):
    """cos(x)"""

    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        return -gy * sin(self.inputs[0])


def cos(x) -> Tensor:
    """Cosine of x."""
    return Cos()(x)


class Tanh(Function):
    """Hyperbolic tangent."""

    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        assert y is not None
        return gy * (1 - y**2)


def tanh(x) -> Tensor:
    """tanh(x)"""
    return Tanh()(x)


class Exp(Function):
    """Exponential function."""

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y


def exp(x) -> Tensor:
    """Exponential of x (e**x)."""
    return Exp()(x)


class Log(Function):
    """Natural logarithm."""

    def forward(self, x):
        return np.log(x + 1e-15)  # avoid log(0)

    def backward(self, gy):
        (x,) = self.inputs
        return gy / x


def log(x) -> Tensor:
    """Natural log of x."""
    return Log()(x)


class MatMul(Function):
    """
    Matrix multiplication x @ W.

    Expects both x and W to be 2D arrays (matrices).
    Does not support batched or higher-dimensional tensor multiplication.
    """

    def forward(self, x, W):
        # use np.matmul to support batched matrix multiplication and higher-dim inputs
        return np.matmul(x, W)

    def backward(self, gy):
        x, W = self.inputs
        # compute gradients using raw numpy on data to avoid recursive MatMul calls
        x_data = x.data
        W_data = W.data
        gy_data = gy.data

        gx_data = np.matmul(gy_data, np.swapaxes(W_data, -1, -2))

        gW_raw = np.matmul(np.swapaxes(x_data, -1, -2), gy_data)
        if W_data is not None and W_data.ndim == 2 and gW_raw.ndim > 2:
            axes = tuple(range(gW_raw.ndim - 2))
            gW_data = gW_raw.sum(axis=axes)
        else:
            gW_data = gW_raw

        from syna.core import as_tensor

        gx = as_tensor(gx_data)
        gW = as_tensor(gW_data)
        return gx, gW


def matmul(x, W) -> Tensor:
    """Matrix multiply x @ W."""
    return MatMul()(x, W)


class Linear(Function):
    """Linear layer y = x.dot(W) + b (b optional)."""

    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        # compute gx = gy @ W.T (supports batched gy)
        gx = matmul(gy, W.T)

        x_data = x.data
        gy_data = gy.data

        gW_raw = np.matmul(np.swapaxes(x_data, -1, -2), gy_data)
        if gW_raw.ndim > 2:
            axes = tuple(range(gW_raw.ndim - 2))
            gW_data = gW_raw.sum(axis=axes)
        else:
            gW_data = gW_raw

        gW = as_tensor(gW_data)
        return gx, gW, gb


def linear(x, W, b=None) -> Tensor:
    """Linear transformation with optional bias."""
    return Linear()(x, W, b)


def linear_simple(x, W, b=None) -> Tensor:
    """A slightly optimized linear for common case: returns t + b when b is given."""
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y


def mean(x, axis: Optional[Tuple[int, ...]] = None, keepdims=False) -> Tensor:
    """Mean like torch.mean: mean over all elements by default, or over given axis/axes."""
    x = as_tensor(x)
    if axis is None:
        denom = float(x.data.size)
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        denom = 1
        for ax in axes:
            denom *= x.shape[ax]
        denom = float(denom)
    return sum(x, axis=axis, keepdims=keepdims) / denom


class Sqrt(Function):
    """Elementwise sqrt with dedicated backward (faster than generic pow)."""

    def forward(self, x):
        return np.sqrt(x)

    def backward(self, gy):
        y = self.outputs[0]()
        assert y is not None
        return gy / (2.0 * y.data)


def sqrt(x) -> Tensor:
    """Elementwise square root."""
    x = as_tensor(x)
    return Sqrt()(x)


class Abs(Function):
    """Elementwise absolute value with direct sign-based backward."""

    def forward(self, x):
        return np.abs(x)

    def backward(self, gy):
        x = self.inputs[0]
        return gy * np.sign(x.data)


def abs(x) -> Tensor:
    """Elementwise absolute value."""
    x = as_tensor(x)
    return Abs()(x)


class Minimum(Function):
    """Elementwise minimum with direct backward using masks (faster than algebraic identity)."""

    def forward(self, x0, x1):
        return np.minimum(x0, x1)

    def backward(self, gy):
        x0, x1 = self.inputs
        mask0 = x0.data <= x1.data
        mask1 = ~mask0
        gx0 = gy * mask0
        gx1 = gy * mask1
        if x0.shape != x1.shape:
            return sum_to(gx0, x0.shape), sum_to(gx1, x1.shape)
        return gx0, gx1


def minimum(x, y) -> Tensor:
    """Elementwise minimum using a dedicated Function for performance."""
    return Minimum()(x, y)


class Maximum(Function):
    """Elementwise maximum with direct backward using masks."""

    def forward(self, x0, x1):
        return np.maximum(x0, x1)

    def backward(self, gy):
        x0, x1 = self.inputs
        mask0 = x0.data >= x1.data
        mask1 = ~mask0
        gx0 = gy * mask0
        gx1 = gy * mask1
        if x0.shape != x1.shape:
            return sum_to(gx0, x0.shape), sum_to(gx1, x1.shape)
        return gx0, gx1


def maximum(x, y) -> Tensor:
    """Elementwise maximum using a dedicated Function for performance."""
    return Maximum()(x, y)


class Max(Function):
    """Max reduction with correct backward distribution."""

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.max(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()
        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = x.data == y.data
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    """Min reduction re-uses Max implementation but calls np.min on forward."""

    def forward(self, x):
        return x.min(axis=self.axis, keepdims=self.keepdims)


def max(x, axis=None, keepdims=False):
    """Max reduction wrapper."""
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    """Min reduction wrapper."""
    return Min(axis, keepdims)(x)


class Clip(Function):
    """Clip values into [x_min, x_max]."""

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        return np.clip(x, self.x_min, self.x_max)

    def backward(self, gy):
        (x,) = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        return gy * mask


def clip(x, x_min, x_max) -> Tensor:
    """Clip tensor values to [x_min, x_max]."""
    return Clip(x_min, x_max)(x)
