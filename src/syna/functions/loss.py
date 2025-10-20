import numpy as np

from syna import utils
from syna.core import Function, Tensor, as_tensor
from syna.functions.activation import softmax
from syna.functions.math import clip, log


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
    return Tensor(acc)
