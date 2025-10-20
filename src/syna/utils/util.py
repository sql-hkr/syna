"""Utilities — helper functions used across syna.

Utility functions used across the syna library: graph plotting, numerical
grad checks, array helpers, convolution dimension helpers, file caching,
and a few tensor-related utilities.
"""

import os
import urllib.request
from typing import Optional, Tuple, Union

import numpy as np

import syna

# --- array/tensor helpers -----------------------------------------------------------


def sum_to(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sum elements of array `x` so that the result has shape `shape`.
    This implements broadcasting-compatible sum reduction.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axes = tuple(range(lead)) if lead > 0 else ()
    axes = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axes + axes, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axes)
    return y


def reshape_sum_backward(
    gy: np.ndarray,
    x_shape: Tuple[int, ...],
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
) -> np.ndarray:
    """
    Reshape `gy` (gradient of a reduced array) to match the original
    input shape `x_shape` before reduction.
    """
    ndim = len(x_shape)
    if axis is None:
        tupled_axis = None
    elif isinstance(axis, int):
        tupled_axis = (axis,)
    else:
        tupled_axis = axis

    if ndim != 0 and tupled_axis is not None and not keepdims:
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    return gy.reshape(shape)


def logsumexp(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Stable log-sum-exp along an axis. Returns array with summed axis kept.
    """
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


def max_backward_shape(
    x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]]
) -> list:
    """
    Compute the shape of gradient for max reduction so the result can be
    broadcast back to `x`.
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    return [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]


# --- numerical gradient / checks ---------------------------------------------------


def gradient_check(
    f, x, *args, rtol: float = 1e-4, atol: float = 1e-5, **kwargs
) -> bool:
    """
    Check gradients of function `f` w.r.t. tensor `x` using numerical approximation.
    `f` should accept and return syna.Tensor. Returns True if gradients match.
    """
    x = syna.as_tensor(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    ok = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)
    if not ok:
        print("\n========== FAILED (Gradient Check) ==========")
        print("Numerical Grad")
        print(" shape:", num_grad.shape)
        print(" values:", str(num_grad.flatten()[:10])[1:-1], "...")
        print("Backprop Grad")
        print(" shape:", bp_grad.shape)
        print(" values:", str(bp_grad.flatten()[:10])[1:-1], "...")
    return ok


def numerical_grad(f, x, *args, **kwargs) -> np.ndarray:
    """
    Compute numerical gradient of `f` at `x` using central differences.
    `x` can be a syna.Tensor or a numpy array.
    """
    eps = 1e-4
    x_arr = x.data if isinstance(x, syna.Tensor) else x
    grad = np.zeros_like(x_arr)

    it = np.nditer(x_arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp = x_arr[idx].copy()

        x_arr[idx] = tmp + eps
        y1 = f(x_arr, *args, **kwargs)
        if isinstance(y1, syna.Tensor):
            y1 = y1.data
        y1 = y1.copy()

        x_arr[idx] = tmp - eps
        y2 = f(x_arr, *args, **kwargs)
        if isinstance(y2, syna.Tensor):
            y2 = y2.data
        y2 = y2.copy()

        grad[idx] = ((y1 - y2).sum()) / (2 * eps)

        x_arr[idx] = tmp
        it.iternext()
    return grad


# --- numpy wrappers -----------------------------------------------------------------


def array_equal(a, b) -> bool:
    """Compare two arrays (or Tensors) for exact equality."""
    a = a.data if isinstance(a, syna.Tensor) else a
    b = b.data if isinstance(b, syna.Tensor) else b
    return np.array_equal(a, b)


def array_allclose(a, b, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Compare two arrays (or Tensors) for approximate equality."""
    a = a.data if isinstance(a, syna.Tensor) else a
    b = b.data if isinstance(b, syna.Tensor) else b
    return np.allclose(a, b, atol=atol, rtol=rtol)


# --- download / cache helpers ------------------------------------------------------


def show_progress(block_num: int, block_size: int, total_size: int):
    """Simple progress bar used by urllib.request.urlretrieve."""
    bar_template = "\r[{}] {:.2f}%"
    downloaded = block_num * block_size
    p = downloaded / total_size * 100 if total_size else 0.0
    i = int(downloaded / total_size * 30) if total_size else 0
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end="")


cache_dir = os.path.join(os.path.expanduser("~"), ".syna")


def get_file(url: str, file_name: Optional[str] = None) -> str:
    """
    Download `url` to the local cache directory and return the local path.
    If file exists in cache, return the cached path without downloading.
    """
    if file_name is None:
        file_name = url[url.rfind("/") + 1 :]
    file_path = os.path.join(cache_dir, file_name)

    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(file_path):
        return file_path

    print("Downloading:", file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")
    return file_path


def pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Ensure `x` is a pair (tuple of two ints)."""
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError("tuple must have length 2")
        return x
    raise ValueError("pair() accepts int or tuple of length 2")
