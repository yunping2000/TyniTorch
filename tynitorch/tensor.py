from typing import Any, Optional, Sequence, Tuple
from uuid import uuid4

from .storage import infer_dtype, make_storage, read_flat, reshape_flat
from .typing import Device, DType, _SHAPE


def compute_default_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    """Compute contiguous row-major strides in units of elements."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class Tensor:
    def __init__(self, data: Any, device: str = "cpu", dtype: Optional[DType] = None):
        self.storage, shape = make_storage(data, device, dtype)
        self.shape = tuple(shape)
        self.strides = compute_default_strides(self.shape)
        self.offset = 0
        self.device = self.storage.device
        self.dtype = self.storage.dtype

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
        flat = read_flat(self.storage, self.shape, self.strides, self.offset)
        nested = reshape_flat(flat, self.shape)
        return _format_nested(nested)


def _format_nested(value: Any, indent: int = 0) -> str:
    """Pretty-format nested lists to resemble an ndarray literal."""
    if not isinstance(value, list):
        return str(value)

    if not value:  # empty list
        return "[]"

    if not isinstance(value[0], list):  # 1D
        return "[" + ", ".join(str(v) for v in value) + "]"

    pad = "  " * indent
    inner_pad = "  " * (indent + 1)
    items = [_format_nested(v, indent + 1) for v in value]
    joined = (",\n".join(inner_pad + item for item in items))
    return "[\n" + joined + "\n" + pad + "]"
