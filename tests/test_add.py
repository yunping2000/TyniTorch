# tests/test_add.py

import pytest
from tynitorch import (
    Tensor,
    DType,
    DeviceType,
    cuda,
)

def test_add_cpu():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu", dtype=DType.FLOAT32)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu", dtype=DType.FLOAT32)
    c = a + b
    assert c.shape == (2, 2)
    assert str(c) == "[\n  [6.0, 8.0],\n  [10.0, 12.0]\n]"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA runtime not available")
def test_add_cuda():
    a = Tensor([[1.0]], device="cuda", dtype=DType.FLOAT32)
    b = Tensor([[1.0]], device="cuda", dtype=DType.FLOAT32)
    c = a + b
    assert c.device.type == DeviceType.CUDA
    assert c.shape == (1, 1)
    assert str(c) == "[\n  [2.0]\n]"


def test_add_transposed_cpu():
    """Addition should work for non-contiguous (transposed) inputs."""
    base = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu", dtype=DType.FLOAT32)
    a = base.transpose(0, 1)  # shape (3, 2), non-contiguous view
    b = Tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], device="cpu", dtype=DType.FLOAT32)

    c = a + b

    assert c.shape == (3, 2)
    assert str(c) == "[\n  [11.0, 24.0],\n  [32.0, 45.0],\n  [53.0, 66.0]\n]"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA runtime not available")
def test_add_transposed_cuda():
    """CUDA add should materialize non-contiguous inputs correctly."""
    base = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=DType.FLOAT32)
    a = base.transpose(0, 1)  # shape (3, 2), non-contiguous view
    b = Tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], device="cuda", dtype=DType.FLOAT32)

    c = a + b

    assert c.device.type == DeviceType.CUDA
    assert c.shape == (3, 2)
    assert str(c) == "[\n  [11.0, 24.0],\n  [32.0, 45.0],\n  [53.0, 66.0]\n]"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA runtime not available")
def test_add_cuda_strided_both_inputs():
    """Both inputs non-contiguous should hit strided CUDA add path."""
    base_a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=DType.FLOAT32)
    base_b = Tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device="cuda", dtype=DType.FLOAT32)
    a = base_a.transpose(0, 1)  # (3, 2)
    b = base_b.transpose(0, 1)  # (3, 2)

    c = a + b

    assert c.device.type == DeviceType.CUDA
    assert c.shape == (3, 2)
    assert str(c) == "[\n  [11.0, 44.0],\n  [22.0, 55.0],\n  [33.0, 66.0]\n]"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA runtime not available")
def test_add_cuda_strided_3d_transpose():
    """3D transpose triggers strided CUDA add and preserves order."""
    base = Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        device="cuda",
        dtype=DType.FLOAT32,
    )
    other = Tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
            [[90.0, 100.0], [110.0, 120.0]],
        ],
        device="cuda",
        dtype=DType.FLOAT32,
    )

    a = base.transpose(0, 2)   # shape (2, 2, 3), non-contiguous
    b = other.transpose(0, 2)  # shape (2, 2, 3), non-contiguous

    c = a + b

    assert c.device.type == DeviceType.CUDA
    assert c.shape == (2, 2, 3)
    assert str(c) == "[\n  [\n    [11.0, 55.0, 99.0],\n    [33.0, 77.0, 121.0]\n  ],\n  [\n    [22.0, 66.0, 110.0],\n    [44.0, 88.0, 132.0]\n  ]\n]"
