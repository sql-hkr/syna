from typing import Optional

import numpy as np

import syna
from syna import utils
from syna.core import Function, Tensor, as_array, as_tensor


class Add(Function):
    def forward(self, x0, x1) -> np.ndarray:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy):
        if self.x0_shape != self.x1_shape:  # for broadcast
            return sum_to(gy, self.x0_shape), sum_to(gy, self.x1_shape)
        return gy, gy


def add(x0, x1) -> Tensor:
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            return sum_to(gx0, x0.shape), sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1) -> Tensor:
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x) -> Tensor:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        if self.x0_shape != self.x1_shape:  # for broadcast
            return sum_to(gy, self.x0_shape), sum_to(-gy, self.x1_shape)
        return gy, -gy


def sub(x0, x1) -> Tensor:
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:  # for broadcast
            return (
                sum_to(gx0, x0.shape),
                sum_to(gx1, x1.shape),
            )
        return gx0, gx1


def div(x0, x1) -> Tensor:
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c) -> None:
        self.c = c

    def forward(self, x):
        return x**self.c

    def backward(self, gy):
        x = self.inputs[0]
        return self.c * x ** (self.c - 1) * gy


def pow(x, c) -> Tensor:
    return Pow(c)(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy * cos(self.inputs[0])


def sin(x) -> Tensor:
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        return -gy * sin(self.inputs[0])


def cos(x) -> Tensor:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        assert y is not None
        return gy * (1 - y**2)


def tanh(x) -> Tensor:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y


def exp(x) -> Tensor:
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        (x,) = self.inputs
        return gy / x


def log(x) -> Tensor:
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape) -> Tensor:
    if x.shape == shape:
        return as_tensor(x)
    return Reshape(shape)(x)


class Transpose(Function):
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
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        return x[self.slices]

    def backward(self, gy):
        (x,) = self.inputs
        return GetItemGrad(self.slices, x.shape)(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, x):
        gx = np.zeros(self.in_shape, dtype=x.dtype)
        np.add.at(gx, self.slices, x)
        return gx


def get_item(x, slices) -> Tensor:
    return GetItem(slices)(x)


def expand_dims(x, axis):
    x = as_tensor(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x, axis: Optional[tuple] = None, keepdims=False) -> Tensor:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)


def sum_to(x, shape: tuple) -> Tensor:
    if x.shape == shape:
        return as_tensor(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)


def broadcast_to(x, shape: tuple) -> Tensor:
    if x.shape == shape:
        return as_tensor(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W) -> Tensor:
    return MatMul()(x, W)


class Linear(Function):
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
    return Linear()(x, W, b)


def linear_simple(x, W, b=None) -> Tensor:
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


def sigmoid_simple(x) -> Tensor:
    x = as_tensor(x)
    return 1 / (1 + exp(-x))


class Sigmoid(Function):
    def forward(self, x):
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)


def sigmoid(x) -> Tensor:
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x) -> Tensor:
    return ReLU()(x)


def softmax_simple(x, axis=1):
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
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
    return Softmax(axis)(x)


def mean_squared_error_simple(x0, x1):
    x0, x1 = as_tensor(x0), as_tensor(x1)
    diff = x0 - x1
    return sum(diff**2) / len(diff)


class MeanSquaredError(Function):
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
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(x, t) -> Tensor:
    x, t = as_tensor(x), as_tensor(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
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
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t) -> Tensor:
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y, t) -> Tensor:
    y, t = as_tensor(y), as_tensor(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()
    return Tensor(as_array(acc))


def dropout(x, dropout_ratio=0.5) -> Tensor:
    x = as_tensor(x)
    if syna.Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        return x * mask / scale
    else:
        return x


class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

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
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max) -> Tensor:
    return Clip(x_min, x_max)(x)


# new
def mean(x) -> Tensor:
    return sum(x) / x.data.shape[0]


def sqrt(x) -> Tensor:
    return x**0.5


def abs(x) -> Tensor:
    return sqrt(x**2)


def minimum(x, y) -> Tensor:
    return 0.5 * (x + y - abs(x - y))
