from syna.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param) -> None:
        param.data -= self.lr * param.grad.data
