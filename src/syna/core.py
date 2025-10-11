"""Core — Tensor and autograd primitives.

Core Tensor, Parameter, Config and autograd Function primitives used across
the syna library. This module contains the lightweight Tensor container and
the Function base class which implement forward/backward for automatic
differentiation.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, Optional

import numpy as np

import syna


class Config:
    """Global config flags affecting backprop and training behavior."""

    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name: str, value: bool):
    """Temporarily set a Config attribute inside a context."""
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def test_mode():
    """Context manager to set train flag to False."""
    return using_config("train", False)


def no_grad():
    """Context manager to disable gradient tracking."""
    return using_config("enable_backprop", False)


class Tensor:
    """Simple Tensor container holding data, gradient and creator Function.

    Most operator behavior delegates to syna.functions.* helpers so this class
    focuses on bookkeeping for autograd.
    """

    __array_priority__ = 200

    def __init__(self, data, name: Optional[str] = None) -> None:
        # normalize scalars/lists to numpy arrays
        if not isinstance(data, np.ndarray) and data is not None:
            data = as_array(data)
        self.data = data
        self.name = name
        self.grad: Optional[Tensor] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "tensor(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "tensor(" + p + ")"

    # arithmetic delegations
    def __add__(self, other):
        return syna.functions.add(self, other)

    def __radd__(self, other):
        return syna.functions.add(self, other)

    def __mul__(self, other):
        return syna.functions.mul(self, other)

    def __rmul__(self, other):
        return syna.functions.mul(self, other)

    def __neg__(self):
        return syna.functions.neg(self)

    def __sub__(self, other):
        return syna.functions.sub(self, other)

    def __rsub__(self, other):
        return syna.functions.sub(other, self)

    def __truediv__(self, other):
        return syna.functions.div(self, other)

    def __rtruediv__(self, other):
        return syna.functions.div(other, self)

    def __pow__(self, other):
        return syna.functions.pow(self, other)

    def __getitem__(self, other):
        return syna.functions.get_item(self, other)

    def max(self, **kwargs):
        return syna.functions.max(self, **kwargs)

    def min(self, **kwargs):
        return syna.functions.min(self, **kwargs)

    def set_creator(self, func: Function) -> None:
        """Mark this tensor as created by func (used for backprop ordering)."""
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self) -> None:
        """Remove reference to creator to break the computational graph."""
        self.creator = None

    def cleargrad(self) -> None:
        """Clear stored gradient."""
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False) -> None:
        """Run backpropagation to compute gradients of inputs.

        Args:
            retain_grad: if False, intermediate gradients are cleared to save memory.
            create_graph: if True, create graph for higher-order gradients.
        """
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))

        funcs: list[Function] = []
        seen_set = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        if self.creator is None:
            return  # nothing to backprop

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            with using_config("enable_backprop", create_graph):
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def unchain_backward(self):
        """Remove creators for all upstream tensors (useful for freeing graph)."""
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape: int):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return syna.functions.reshape(self, shape)

    def transpose(self, *axes: int):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return syna.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return syna.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        return syna.functions.transpose(self)


class Parameter(Tensor):
    """A thin wrapper for trainable parameters (keeps API separate)."""

    pass


def as_tensor(obj) -> Tensor:
    """Ensure obj is a Tensor; convert scalars/arrays to Tensor if needed."""
    if isinstance(obj, Tensor):
        return obj
    return Tensor(as_array(obj))


def as_array(x) -> np.ndarray:
    """Convert scalars to numpy arrays; leave arrays unchanged."""
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """Base Function for forward/backward ops used in autograd.

    Subclasses should implement forward (numpy arrays -> numpy arrays) and
    backward (Tensor grads -> Tensor grads).
    """

    def __call__(self, *input: Tensor | np.ndarray | int | float) -> Any:
        inputs = [as_tensor(x) for x in input]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [as_tensor(y) for y in ys]

        if Config.enable_backprop:
            self.generation = max(x.generation for x in inputs) if inputs else 0
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *args: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, *args: Tensor) -> Tensor | list[Tensor]:
        raise NotImplementedError()
