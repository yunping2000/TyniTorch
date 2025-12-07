import pytest

from tynitorch import DType, Tensor
from tynitorch.typing import DeviceType


def test_tensor_create_cpu():
    tensor = Tensor([[1, 2], [3, 4]], device="cpu", dtype=DType.INT64)
    assert tensor.shape == (2, 2)
    assert tensor.device.type == DeviceType.CPU
    assert tensor.dtype == DType.INT64
    assert str(tensor) == "[\n  [1, 2],\n  [3, 4]\n]"


def test_tensor_create_cuda():
    try:
        tensor = Tensor([[1.0]], device="cuda", dtype=DType.FLOAT32)
    except NotImplementedError:
        pytest.skip("CUDA runtime is not available")
    assert tensor.device.type == DeviceType.CUDA
    assert tensor.dtype == DType.FLOAT32
    assert tensor.shape == (1, 1)
