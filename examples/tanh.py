"""
Need the dot binary from the graphviz package (www.graphviz.org).
"""

import syna
import syna.functions as F
from syna.utils import plot_dot_graph

x = syna.tensor(1.0)
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters = 4

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file="tanh.svg")
