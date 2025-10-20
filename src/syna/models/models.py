"""Model utilities and common model classes.

Provides a Model base class (a Layer with plotting helpers) and a simple
MLP implementation used in examples and tests.
"""

import syna.functions as F
from syna import utils
from syna.layers import Layer, Linear


class Model(Layer):
    """Base model class (also a Layer) with a convenience plot method.

    The plot method runs forward on the provided inputs and writes a DOT
    graph of the computation to the given file.
    """

    def plot(self, *inputs, to_file="model.png") -> None:
        """Run forward and save a DOT graph of the output computation."""
        y = self.forward(*inputs)
        utils.viz.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    """Simple multi-layer perceptron.

    Args:
        fc_output_sizes (iterable[int]): Sizes for successive Linear layers.
        activation (callable): Activation applied between layers (not after
            the final layer). Defaults to F.sigmoid.
    """

    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        # create Linear layers and expose them as attributes l0, l1, ...
        self.layers = [Linear(size) for size in fc_output_sizes]
        for i, layer in enumerate(self.layers):
            setattr(self, f"l{i}", layer)

    def forward(self, x):
        """Forward pass: apply activation after every layer except the last."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
