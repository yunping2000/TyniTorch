from typing import Any, Optional, Sequence, Tuple
from uuid import uuid4

import ctypes

from .storage import (
    Storage,
    reshape_flat,
)
from .typing import Device, DeviceType, DType, DTYPE_SIZES, _SHAPE

def infer_dtype(data: Any) -> DType:
    has_float = False
    def _walk(x: Any) -> None:
        nonlocal has_float
        if isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)
        elif isinstance(x, bool):
            return
        elif isinstance(x, float):
            has_float = True
        elif isinstance(x, int):
            return
        else:
            raise TypeError(f"Unsupported data type: {type(x)}")
    _walk(data)
    return DType.FLOAT32 if has_float else DType.INT64


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        child_shape = _infer_shape(data[0])
        for item in data:
            if _infer_shape(item) != child_shape:
                raise ValueError("Inconsistent shapes in nested data")
        return (len(data),) + child_shape
    else:
        return ()


def infer_shape(data: Any) -> Tuple[int, ...]:
    return _infer_shape(data)


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
        self.shape = infer_shape(data) # May raise exception if data is malformed

        self.dtype = dtype if dtype is not None else infer_dtype(data)
        self.storage = Storage.allocate(self.num_elements(), self.dtype, device)
        self.storage.copy_from_list(data)

        self.strides = compute_default_strides(self.shape)
        self.offset = 0
        self.device = self.storage.device
        self.uuid = uuid4()
        self.grad = None

    def num_elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @classmethod
    def from_storage(
        cls,
        storage: Storage,
        shape: Sequence[int],
        strides: Optional[Sequence[int]] = None,
        offset: Optional[int] = None,
    ) -> "Tensor":
        """Create a Tensor that wraps an existing Storage."""
        shape_tuple = tuple(shape)
        if strides is None:
            strides = compute_default_strides(shape_tuple)
        strides_tuple = tuple(strides)

        tensor = cls.__new__(cls)
        tensor.storage = storage
        tensor.shape = shape_tuple
        tensor.strides = strides_tuple
        tensor.offset = 0 if offset is None else offset
        tensor.device = storage.device
        tensor.dtype = storage.dtype
        tensor.uuid = uuid4()
        tensor.grad = None
        return tensor


    """
    Return a new Tensor that is a transposed view of this tensor along the given dimensions.
    No data is copied; the new tensor shares the same storage as the original.
    """
    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        if dim0 < 0 or dim1 < 0 or dim0 >= len(self.shape) or dim1 >= len(self.shape):
            raise IndexError(f"Dimension out of range, got dim0={dim0}, dim1={dim1} for tensor with shape {self.shape}")

        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        tensor = Tensor.from_storage(
            storage=self.storage,
            shape=new_shape,
            strides=new_strides,
            offset=self.offset,
        )
        return tensor


    """
    Return a new Tensor that has a different shape but shares the same data.
    The total number of elements must remain the same.
    """
    def view(self, shape: _SHAPE) -> "Tensor":
        if not self.is_contiguous():
            raise ValueError("Can only view a contiguous tensor.")

        total_elements_old = 1
        for dim in self.shape:
            total_elements_old *= dim

        total_elements_new = 1
        for dim in shape:
            total_elements_new *= dim

        if total_elements_old != total_elements_new:
            raise ValueError(f"Cannot view tensor of shape {self.shape} as shape {shape} due to mismatch in number of elements.")

        tensor = Tensor.from_storage(
            storage=self.storage,
            shape=shape,
            strides=compute_default_strides(shape),
            offset=self.offset,
        )
        return tensor


    """
    Return a contiguous copy of the tensor. Data may be copied if the tensor is not currently contiguous.
    """
    def contiguous(self) -> "Tensor":
        if self.is_contiguous():
            return self

        from . import ops
        return ops.contiguous(self)

    def is_contiguous(self) -> bool:
        expected_stride = 1
        for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
            if stride != expected_stride:
                return False
            expected_stride *= dim
        return True


    def __add__(self, other: "Tensor") -> "Tensor":
        # Import ops here to avoid circular import issues
        from . import ops
        return ops.add(self, other)


    def __repr__(self):
        flat = self.storage.read_flat(self.shape, self.strides, self.offset)
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
