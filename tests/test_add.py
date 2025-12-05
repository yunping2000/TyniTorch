# tests/test_add.py

from tynitorch import Tensor
from tynitorch import DType


def test_add_cpu():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu", dtype=DType.FLOAT32)
    c = a + b
    assert c.shape == (2, 2)
    assert str(c) == "[\n  [6.0, 8.0],\n  [10.0, 12.0]\n]"


def test_add_cuda():
    # Placeholder until CUDA backend is implemented
    try:
        Tensor([[1.0]], device="cuda", dtype=DType.FLOAT32)
    except NotImplementedError:
        return
