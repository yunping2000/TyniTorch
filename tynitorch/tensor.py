import torch
from typing import Any
from uuid import uuid4

from .torch_helper import dtype_tyni_2_torch, dtype_torch_2_tyni
from .storage import make_storage
from .typing import dtype

class Tensor:
    def __init__(self, data: Any, device: str="cpu", dtype: dtype=None):
        try:
            data_torch = torch.tensor(data, dtype=dtype_tyni_2_torch(dtype))
        except ValueError as e:
            raise ValueError(f"Data provided cannot be converted to a torch tensor : {e}")

        self.storage = make_storage(data_torch, device)
        self.shape = tuple(data_torch.shape)
        self.device = device
        self.dtype = dtype if dtype is not None else dtype_torch_2_tyni(data_torch)

        self.uuid = uuid4()

        self.grad = None

    def __add__(self, other: "Tensor") -> "Tensor":
        # Import ops here to avoid circular import issues
        from . import ops
        return ops.add(self, other)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device={self.device}, uuid={self.uuid})"