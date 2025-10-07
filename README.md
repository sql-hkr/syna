# Syna

![GitHub Release](https://img.shields.io/github/v/release/sql-hkr/syna)
![GitHub License](https://img.shields.io/github/license/sql-hkr/syna)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/sql-hkr/syna/ci.yml?label=CI)



Syna is a lightweight machine learning framework inspired by [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3). Built from scratch using only NumPy, it follows a define-by-run (dynamic computation graph) approach and includes a basic reinforcement learning framework.

Unlike most frameworks that implement reinforcement learning as a separate library, Syna provides everything in a single library.

Designed for beginners and researchers, Syna helps you learn the fundamentals of machine learning and the inner workings of frameworks like [PyTorch](https://github.com/pytorch/pytorch). Performance is not the focus, and GPU support is intentionally omitted to keep the code simple and easy to understand.


## Installation

```bash
git clone https://github.com/sql-hkr/syna.git
cd syna
uv venv
source .venv/bin/activate
uv sync
```

## License

Syna is licensed under the MIT License. See [LICENSE](LICENSE) for details.

