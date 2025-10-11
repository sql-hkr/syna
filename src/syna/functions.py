"""Functions — differentiable operations for autograd.

Core differentiable functions (forward/backward) used by the syna framework.
This module provides NumPy-backed implementations of common operations (arithmetic,
matrix ops, activations, reductions, and losses) as Function subclasses so they
can participate in automatic differentiation.
"""

from typing import Optional, Tuple

import numpy as np

import syna
from syna import utils
from syna.core import Function, Tensor, as_array, as_tensor


# ----------------------
# Arithmetic operations
# ----------------------
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
        return np.log(x)

    def backward(self, gy):
        (x,) = self.inputs
        return gy / x


def log(x) -> Tensor:
    """Natural log of x."""
    return Log()(x)


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


# ----------------------
# Linear algebra
# ----------------------
class MatMul(Function):
    """Matrix multiplication x.dot(W)."""

    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
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
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
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


# ----------------------
# Activations
# ----------------------
def sigmoid_simple(x) -> Tensor:
    """Sigmoid implemented via exp; returns as Tensor."""
    x = as_tensor(x)
    return 1 / (1 + exp(-x))


class Sigmoid(Function):
    """Numerically stable tanh-based sigmoid."""

    def forward(self, x):
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)


def sigmoid(x) -> Tensor:
    """Sigmoid activation."""
    return Sigmoid()(x)


class ReLU(Function):
    """Rectified Linear Unit."""

    def forward(self, x):
        return np.maximum(x, 0.0)

    def backward(self, gy):
        (x,) = self.inputs
        mask = x.data > 0
        return gy * mask


def relu(x) -> Tensor:
    """ReLU activation."""
    return ReLU()(x)


def softmax_simple(x, axis=1):
    """Softmax using safe exp/normalization helpers."""
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    """Softmax with stable forward and correct backward."""

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        return y / y.sum(axis=self.axis, keepdims=True)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy[0]
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        return gx - y * sumdx


def softmax(x, axis=1) -> Tensor:
    """Softmax along specified axis."""
    return Softmax(axis)(x)


# ----------------------
# Losses & metrics
# ----------------------
def mean_squared_error_simple(x0, x1):
    """Simple mean squared error (for convenience)."""
    x0, x1 = as_tensor(x0), as_tensor(x1)
    diff = x0 - x1
    return sum(diff**2) / len(diff)


class MeanSquaredError(Function):
    """Mean squared error over the batch."""

    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff**2).sum() / len(diff)

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1) -> Tensor:
    """Mean squared error between x0 and x1."""
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    """Softmax cross entropy (expects integer class labels t)."""

    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape
        gy *= 1 / N
        y = softmax(x)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        return (y - t_onehot) * gy


def softmax_cross_entropy(x, t) -> Tensor:
    """Softmax cross entropy between logits x and labels t."""
    return SoftmaxCrossEntropy()(x, t)


def softmax_cross_entropy_simple(x, t) -> Tensor:
    """Simple softmax cross entropy using explicit softmax and log clipping."""
    x, t = as_tensor(x), as_tensor(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    return -sum(tlog_p) / N


def accuracy(y, t) -> Tensor:
    """Compute classification accuracy: mean(pred == t)."""
    y, t = as_tensor(y), as_tensor(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    acc = (pred == t.data).mean()
    return Tensor(as_array(acc))


# ----------------------
# Misc
# ----------------------
def dropout(x, dropout_ratio=0.5) -> Tensor:
    """Dropout during training; identity during evaluation."""
    x = as_tensor(x)
    if syna.Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        return x * mask / scale
    return x


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


# Convenience aliases and elementwise ops built from primitives
def mean(x) -> Tensor:
    """Mean over the first dimension (batch mean)."""
    return sum(x) / x.data.shape[0]


def sqrt(x) -> Tensor:
    """Elementwise square root."""
    return x**0.5


def abs(x) -> Tensor:
    """Elementwise absolute value."""
    return sqrt(x**2)


def minimum(x, y) -> Tensor:
    """Elementwise minimum using algebraic identity to enable autograd."""
    return 0.5 * (x + y - abs(x - y))
