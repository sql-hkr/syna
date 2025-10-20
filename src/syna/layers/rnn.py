from typing import Optional

import syna.functions as F
from syna.layers.layer import Layer, Linear


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
