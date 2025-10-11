"""
Neural network layer definitions.

Defines the Layer base class and common layer implementations such as
Linear, RNN and LSTM. Layers manage parameters and support saving/loading
weights.
"""

import os
import weakref
from typing import Dict, Optional

import numpy as np

import syna.functions as F
from syna.core import Parameter


class Layer:
    """
    Base layer class that tracks Parameters and sub-Layers.

    Subclasses must implement forward(). Layers register any attribute
    that is a Parameter or Layer automatically.
    """

    def __init__(self) -> None:
        self._params = set()

    def __setattr__(self, name, value):
        # register parameters and sub-layers
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *args):
        """
        Run forward and store weak references to outputs for bookkeeping.
        Returns a single value when forward returns a single output.
        """
        outputs = self.forward(*args)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # keep weak refs to outputs (for memory-friendly graph references)
        self.inputs = [weakref.ref(x) for x in outputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        """Compute the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError()

    def params(self):
        """Yield all Parameter objects in this layer (including nested layers)."""
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        """Clear gradients of all parameters."""
        for param in self.params():
            param.cleargrad()

    def _flatten_params(self, params_dict: Dict[str, Parameter], parent_key: str = ""):
        """Populate params_dict with flattened parameter names -> Parameter."""
        for name in self._params:
            obj = self.__dict__[name]
            key = f"{parent_key}/{name}" if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path: str):
        """Save layer parameters to a compressed .npz file."""
        params_dict: Dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        array_dict = {k: p.data for k, p in params_dict.items() if p is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except Exception:
            # if writing failed, remove partial file
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path: str):
        """Load parameters from a .npz file created by save_weights()."""
        npz = np.load(path)
        params_dict: Dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    """
    Fully-connected linear layer.

    Args:
      out_size: output dimension
      nobias: if True, no bias is used
      dtype: numpy dtype for parameters
      in_size: optional input dimension; if not provided, W is initialized
               lazily on the first forward pass based on input shape.
    """

    def __init__(
        self,
        out_size: int,
        nobias: bool = False,
        dtype=np.float32,
        in_size: Optional[int] = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()
        self.b = (
            None if nobias else Parameter(np.zeros(out_size, dtype=dtype), name="b")
        )

    def _init_W(self) -> None:
        in_size, out_size = self.in_size, self.out_size
        W_data = np.random.randn(in_size, out_size).astype(self.dtype) * np.sqrt(
            1 / in_size
        )
        self.W.data = W_data

    def forward(self, inputs):
        """Apply linear transformation to inputs. Initializes W lazily if needed."""
        if self.W.data is None:
            self.in_size = inputs.shape[1]
            self._init_W()
        return F.linear(inputs, self.W, self.b)


class RNN(Layer):
    """
    Simple recurrent layer with tanh activation.

    The recurrence is:
      h_t = tanh(x2h(x_t) + h2h(h_{t-1}))  (h2h has no bias)
    """

    def __init__(self, hidden_size: int, in_size: Optional[int] = None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h = None

    def reset_state(self):
        """Reset the hidden state between sequences."""
        self.h = None

    def forward(self, x):
        """Process one timestep (or a batch) and return new hidden state."""
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    """
    LSTM layer implementation.

    Uses separate Linear layers for input-to-gate and hidden-to-gate
    transforms. Hidden-to-gate transforms have no bias.
    """

    def __init__(self, hidden_size: int, in_size: Optional[int] = None):
        super().__init__()
        H, in_size_arg = hidden_size, in_size
        # input -> gates
        self.x2f = Linear(H, in_size=in_size_arg)
        self.x2i = Linear(H, in_size=in_size_arg)
        self.x2o = Linear(H, in_size=in_size_arg)
        self.x2u = Linear(H, in_size=in_size_arg)
        # hidden -> gates (no bias)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.h2u = Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        """Reset hidden and cell states."""
        self.h = None
        self.c = None

    def forward(self, x):
        """
        Compute one step of LSTM.

        Returns the new hidden state h_t.
        """
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        c_new = i * u if self.c is None else (f * self.c) + (i * u)
        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new
