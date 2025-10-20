import numpy as np

from syna.core import Function, Tensor, as_tensor
from syna.functions.function import sum
from syna.functions.math import exp


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
