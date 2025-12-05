import torch
from typing import Any, Optional
from uuid import uuid4

from .torch_helper import dtype_tyni_2_torch, dtype_torch_2_tyni
from .storage import make_storage
from .typing import Device, DType, _SHAPE


class Tensor:
    def __init__(self, data: Any, device: str = "cpu", dtype: Optional[DType] = None):
        device_obj = Device.from_value(device)
        torch_dtype = dtype_tyni_2_torch(dtype) if dtype is not None else None

        try:
            data_torch = torch.tensor(data, dtype=torch_dtype, device=str(device_obj))
        except ValueError as e:
            raise ValueError(f"Data provided cannot be converted to a torch tensor : {e}")

        self.storage = make_storage(data_torch, device)
        self.shape = tuple(data_torch.shape)
        self.strides = None
        self.device = self.storage.device
        self.dtype = dtype if dtype is not None else dtype_torch_2_tyni(data_torch)

        self.uuid = uuid4()

        self.grad = None

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        pass

    def view(self, shape: _SHAPE) -> "Tensor":
        pass

    def __add__(self, other: "Tensor") -> "Tensor":
        # Import ops here to avoid circular import issues
        from . import ops
        return ops.add(self, other)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device={self.device}, uuid={self.uuid})"
