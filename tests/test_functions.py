import unittest

import numpy as np

import syna
import syna.functions as F

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for these tests")
class TestSynaExtra(unittest.TestCase):
    def assert_close(self, a, b, rtol=1e-4, atol=1e-5):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    def _syna_op(self, name, *args, **kwargs):
        # Try function in F first, then method on Tensor, then raise
        if hasattr(F, name):
            return getattr(F, name)(*args, **kwargs)
        # method on first Tensor argument
        if args and isinstance(args[0], syna.tensor) and hasattr(args[0], name):
            return getattr(args[0], name)(*args[1:], **kwargs)
        raise AttributeError(f"operation {name} not found in syna.functions or Tensor")

    def test_unary_ops_grad(self):
        # exp, log, sigmoid, tanh gradients
        N, D = 4, 5
        x_np = np.random.randn(N, D).astype(np.float32) + 2.0  # shift to avoid log(<=0)

        ops = [
            ("exp", lambda t: torch.exp(t)),
            ("log", lambda t: torch.log(t)),
            ("sigmoid", lambda t: torch.sigmoid(t)),
            ("tanh", lambda t: torch.tanh(t)),
        ]

        for name, torch_fn in ops:
            with self.subTest(op=name):
                # Require implementation
                assert hasattr(F, name) or hasattr(syna.tensor, name), (
                    f"{name} not implemented"
                )

                tx = torch.tensor(x_np, requires_grad=True)
                torch_y = torch_fn(tx)
                torch_loss = torch_y.sum()
                torch_loss.backward()

                sx = syna.tensor(x_np)
                syna_y = self._syna_op(name, sx)
                syna_loss = syna_y.sum()
                syna_loss.backward()

                self.assert_close(syna_y.data, torch_y.detach().cpu().numpy())
                self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())

    def test_binary_ops_grad(self):
        # add, sub, mul, div, power (scalar)
        N, D = 3, 4
        a_np = np.random.randn(N, D).astype(np.float32)
        b_np = (
            np.random.randn(N, D).astype(np.float32) + 0.5
        )  # avoid div by zero in some cases

        binary_ops = [
            ("add", lambda x, y: x + y),
            ("sub", lambda x, y: x - y),
            ("mul", lambda x, y: x * y),
            ("div", lambda x, y: x / y),
        ]

        for name, torch_fn in binary_ops:
            with self.subTest(op=name):
                tx = torch.tensor(a_np, requires_grad=True)
                ty = torch.tensor(b_np, requires_grad=True)
                torch_out = torch_fn(tx, ty)
                torch_loss = (torch_out**2).sum()
                torch_loss.backward()

                sx = syna.tensor(a_np)
                sy = syna.tensor(b_np)
                # prefer F.func if exists else operator
                if hasattr(F, name):
                    syna_out = getattr(F, name)(sx, sy)
                else:
                    # use python operator mapping
                    op_map = {
                        "add": sx.__add__,
                        "sub": sx.__sub__,
                        "mul": sx.__mul__,
                        "div": sx.__truediv__,
                    }
                    assert name in op_map, f"{name} not implemented"
                    syna_out = op_map[name](sy)

                syna_loss = (syna_out**2).sum()
                syna_loss.backward()

                self.assert_close(syna_out.data, torch_out.detach().cpu().numpy())
                self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())
                self.assert_close(sy.grad.data, ty.grad.detach().cpu().numpy())

        # power (scalar exponent)
        exp = 3.0
        tx = torch.tensor(a_np, requires_grad=True)
        torch_out = tx**exp
        torch_loss = torch_out.sum()
        torch_loss.backward()

        sx = syna.tensor(a_np)
        if hasattr(F, "power"):
            syna_out = F.power(sx, exp)
        else:
            # try operator; will raise if not implemented
            syna_out = sx**exp

        syna_loss = syna_out.sum()
        syna_loss.backward()

        self.assert_close(syna_out.data, torch_out.detach().cpu().numpy())
        self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())

    def test_matmul_transpose_grad(self):
        # matmul (2D) and transpose gradients
        N, D_in, D_out = 5, 6, 4
        x_np = np.random.randn(N, D_in).astype(np.float32)
        W_np = np.random.randn(D_in, D_out).astype(np.float32)

        tx = torch.tensor(x_np, requires_grad=True)
        tW = torch.tensor(W_np, requires_grad=True)
        torch_out = tx.matmul(tW)
        torch_loss = torch_out.sum()
        torch_loss.backward()

        sx = syna.tensor(x_np)
        sW = syna.tensor(W_np)
        # try F.matmul or operator @
        if hasattr(F, "matmul"):
            syna_out = F.matmul(sx, sW)
        else:
            syna_out = sx @ sW

        syna_loss = syna_out.sum()
        syna_loss.backward()

        self.assert_close(syna_out.data, torch_out.detach().cpu().numpy())
        self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())
        self.assert_close(sW.grad.data, tW.grad.detach().cpu().numpy())

        # transpose behavior: compare transpose of output
        torch_t = torch_out.t()
        # syna transpose: try attribute T or function transpose
        if hasattr(syna_out, "T"):
            syna_t = syna_out.T
        elif hasattr(F, "transpose"):
            syna_t = F.transpose(syna_out)
        elif hasattr(syna_out, "transpose"):
            syna_t = syna_out.transpose()
        else:
            raise AssertionError("transpose not implemented")
        self.assert_close(syna_t.data, torch_t.detach().cpu().numpy())

    def test_reductions_grad(self):
        # sum and mean along axes and full reduction
        N, D = 4, 5
        x_np = np.random.randn(N, D).astype(np.float32)

        tx = torch.tensor(x_np, requires_grad=True)
        torch_sum = tx.sum()
        torch_sum.backward()
        sx = syna.tensor(x_np)
        syna_sum = sx.sum()
        syna_sum.backward()
        self.assert_close(syna_sum.data, torch_sum.detach().cpu().numpy())
        self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())

        # mean
        tx = torch.tensor(x_np, requires_grad=True)
        torch_mean = tx.mean()
        torch_mean.backward()
        sx = syna.tensor(x_np)
        # try F.mean or Tensor.mean
        if hasattr(F, "mean"):
            syna_mean = F.mean(sx)
        elif hasattr(sx, "mean"):
            syna_mean = sx.mean()
        else:
            raise AssertionError("mean not implemented")
        syna_mean.backward()
        self.assert_close(syna_mean.data, torch_mean.detach().cpu().numpy())
        self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())

        # sum along axis 0
        tx = torch.tensor(x_np, requires_grad=True)
        torch_axis_sum = tx.sum(0)
        torch_axis_sum.sum().backward()
        sx = syna.tensor(x_np)
        # prefer function in F then method
        if hasattr(F, "sum"):
            syna_axis_sum = F.sum(sx, axis=0)
        elif hasattr(sx, "sum"):
            syna_axis_sum = sx.sum(0)
        else:
            raise AssertionError("sum(axis) not implemented")
        syna_axis_sum.sum().backward()
        self.assert_close(syna_axis_sum.data, torch_axis_sum.detach().cpu().numpy())
        self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())

    def test_softmax_and_crossentropy_consistency(self):
        # compare softmax outputs and cross entropy combined vs torch (if both exist)
        N, C = 6, 5
        logits = np.random.randn(N, C).astype(np.float32)
        targets = np.random.randint(0, C, size=(N,)).astype(np.int64)

        tx = torch.tensor(logits, requires_grad=True)
        torch_soft = torch.softmax(tx, dim=1)
        # use cross_entropy to get loss and grad
        tt = torch.tensor(targets, dtype=torch.long)
        torch_loss = torch.nn.functional.cross_entropy(tx, tt)
        torch_loss.backward()

        sx = syna.tensor(logits)
        st = syna.tensor(targets)
        # softmax (probabilities) required
        assert hasattr(F, "softmax"), "softmax not implemented"
        syna_soft = F.softmax(sx)
        # ensure dims: compare probs
        self.assert_close(syna_soft.data, torch_soft.detach().cpu().numpy())

        # cross entropy combined (logits -> loss) required
        assert hasattr(F, "softmax_cross_entropy"), (
            "softmax_cross_entropy not implemented"
        )
        syna_loss = F.softmax_cross_entropy(sx, st)
        syna_loss.backward()
        self.assert_close(syna_loss.data, torch_loss.detach().cpu().numpy())
        self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())

    def test_broadcasting_ops_grad(self):
        # Test broadcasting behavior for binary ops: shapes like (N,1) + (1,D), (N,D) + (D,), and scalar
        N, D = 4, 5

        # base arrays
        a = np.random.randn(N, 1).astype(np.float32)
        b = np.random.randn(1, D).astype(np.float32)
        c = np.random.randn(D).astype(np.float32)
        scalar = float(np.random.randn())

        patterns = [
            (a, b),  # (N,1) + (1,D) -> (N,D)
            (np.random.randn(N, D).astype(np.float32), c),  # (N,D) + (D,) -> (N,D)
            (np.random.randn(N, D).astype(np.float32), scalar),  # scalar broadcast
        ]

        ops = [
            ("add", lambda x, y: x + y),
            ("sub", lambda x, y: x - y),
            ("mul", lambda x, y: x * y),
            ("div", lambda x, y: x / y),
        ]

        for left_np, right_np in patterns:
            for name, torch_fn in ops:
                with self.subTest(
                    op=name,
                    left=left_np.shape,
                    right=getattr(right_np, "shape", "scalar"),
                ):
                    # prepare torch tensors
                    tx = torch.tensor(left_np, requires_grad=True)
                    # right may be scalar
                    if np.isscalar(right_np):
                        ty = torch.tensor(right_np, requires_grad=True)
                    else:
                        ty = torch.tensor(right_np, requires_grad=True)

                    torch_out = torch_fn(tx, ty)
                    torch_loss = (torch_out**2).sum()
                    torch_loss.backward()

                    sx = syna.tensor(left_np)
                    sy = syna.tensor(right_np)
                    # get syna op
                    if hasattr(F, name):
                        syna_out = getattr(F, name)(sx, sy)
                    else:
                        op_map = {
                            "add": sx.__add__,
                            "sub": sx.__sub__,
                            "mul": sx.__mul__,
                            "div": sx.__truediv__,
                        }
                        assert name in op_map, (
                            f"{name} not implemented for broadcast pattern"
                        )
                        syna_out = op_map[name](sy)

                    syna_loss = (syna_out**2).sum()
                    syna_loss.backward()

                    # compare outputs
                    self.assert_close(syna_out.data, torch_out.detach().cpu().numpy())

                    # compare gradients: shapes must match original operands
                    # left
                    self.assert_close(sx.grad.data, tx.grad.detach().cpu().numpy())
                    # right (if scalar, torch gives a single value)
                    self.assert_close(sy.grad.data, ty.grad.detach().cpu().numpy())
