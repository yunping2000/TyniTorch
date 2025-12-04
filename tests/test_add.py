# tests/test_add.py

import torch
from tynitorch import Tensor
from tynitorch import dtype


def test_add_cpu():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=dtype.FLOAT32)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu", dtype=dtype.FLOAT32)
    c = a + b
    assert c.shape == (2, 2)
    assert torch.allclose(c.storage.impl, torch.tensor([[6.0, 8.0], [10.0, 12.0]], dtype=torch.float32))


def test_add_cuda():
    if not torch.cuda.is_available():
        return

    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype.FLOAT32)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype.FLOAT32)
    c = a + b

    assert c.shape == a.shape
    assert c.device == "cuda"
    assert torch.allclose(c.storage.impl, a.storage.impl + b.storage.impl)
