"""Functions — differentiable operations for autograd.

Core differentiable functions (forward/backward) used by the syna framework.
This module provides NumPy-backed implementations of common operations (arithmetic,
matrix ops, activations, reductions, and losses) as Function subclasses so they
can participate in automatic differentiation.
"""

from syna.functions.activation import (
    relu,
    sigmoid,
    sigmoid_simple,
    softmax,
    softmax_simple,
)
from syna.functions.function import (
    broadcast_to,
    dropout,
    expand_dims,
    flatten,
    get_item,
    reshape,
    sum,
    sum_to,
    transpose,
)
from syna.functions.loss import (
    accuracy,
    mean_squared_error,
    mean_squared_error_simple,
    softmax_cross_entropy,
    softmax_cross_entropy_simple,
)
from syna.functions.math import (
    abs,
    add,
    clip,
    cos,
    div,
    exp,
    linear,
    linear_simple,
    log,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mul,
    neg,
    pow,
    sin,
    sqrt,
    sub,
    tanh,
)
