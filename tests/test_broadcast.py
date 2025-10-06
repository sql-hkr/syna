import unittest

import numpy as np

import syna.functions as F
from syna import Tensor


class TestBroadcast(unittest.TestCase):
    def test_shape_check(self):
        x = Tensor(np.random.randn(1, 10))
        b = Tensor(np.random.randn(10))
        y = F.add(x, b)
        loss = F.sum(y)
        loss.backward()
        self.assertEqual(b.grad.shape, b.shape)
