# Syna

![PyPI - Version](https://img.shields.io/pypi/v/syna)
![GitHub License](https://img.shields.io/github/license/sql-hkr/syna)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/syna)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/sql-hkr/syna/ci.yml?label=CI)


Syna is a lightweight machine learning framework inspired by [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3). Built from scratch using only NumPy, it follows a define-by-run (dynamic computation graph) approach and includes a basic reinforcement learning framework.

Unlike most frameworks that implement reinforcement learning as a separate library, Syna provides everything in a single library.

Designed for beginners and researchers, Syna helps you learn the fundamentals of machine learning and the inner workings of frameworks like [PyTorch](https://github.com/pytorch/pytorch). Performance is not the focus, and GPU support is intentionally omitted to keep the code simple and easy to understand.


## Installation

Get the Syna Source

```bash
git clone https://github.com/sql-hkr/syna.git
cd syna
uv venv
source .venv/bin/activate
uv sync
```

Or, from [PyPI](https://pypi.org/project/syna/):

```bash
uv add syna
```

> [!IMPORTANT]
> To visualize the computation graph, you need to install [Graphviz](https://graphviz.org).
> ```bash
> brew install graphviz # macOS
> sudo apt install graphviz # Linux
> ```

## Getting started
### Computation Graph

Visualize the computation graph for the fifth derivative of tanh(x) with respect to x.

```python
import syna
import syna.functions as F
from syna import utils

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
utils.viz.plot_dot_graph(gx, verbose=False, to_file="tanh.svg")
```

The output graph is shown below.

![](/docs/_static/graph/tanh.svg)

### RL (Deep Q-Network; DQN)

Solve CartPole-v1 using the DQN algorithm.

```python
from syna.algo.dqn import DQNAgent
from syna.utils.rl import Trainer

trainer = Trainer(
    env_name="CartPole-v1",
    num_episodes=300,
    agent=DQNAgent(
        lr=3e-4,
    ),
)
trainer.train()
```

## API Reference
- [syna package](https://sql-hkr.github.io/syna/api/syna.html)

## License

Syna is licensed under the MIT License. See [LICENSE](LICENSE) for details.

