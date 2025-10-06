import syna
import syna.functions as F
import syna.layers as L
from syna import utils


class Model(syna.layers.Layer):
    def plot(self, *inputs, to_file="model.png") -> None:
        y = self.forward(*inputs)
        utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, inputs):
        for l in self.layers[:-1]:
            inputs = self.activation(l(inputs))
        return self.layers[-1](inputs)
