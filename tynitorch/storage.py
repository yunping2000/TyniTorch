import ctypes
import struct
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .typing import Device, DeviceType, DType, DTYPE_SIZES

# struct format strings for packing/unpacking raw bytes
_STRUCT_FORMATS = {
    DType.FLOAT32: "<f",
    DType.FLOAT64: "<d",
    DType.INT32: "<i",
    DType.INT64: "<q",
}


@dataclass
class Storage:
    # Raw pointer to the allocated memory. For CPU this points into `cpu_buffer`.
    data_ptr: int
    bytes: int
    dtype: DType
    device: Device
    cpu_buffer: Optional[bytearray] = None


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


def _flatten(data: Any, out: List[Any]) -> None:
    if isinstance(data, (list, tuple)):
        for item in data:
            _flatten(item, out)
    else:
        out.append(data)


def infer_shape(data: Any) -> Tuple[int, ...]:
    return _infer_shape(data)


def flatten_data(data: Any) -> List[Any]:
    flat: List[Any] = []
    _flatten(data, flat)
    return flat


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


def reshape_flat(flat: Sequence[Any], shape: Sequence[int]) -> Any:
    if not shape:
        return flat[0] if flat else None

    total = 1
    for dim in shape:
        total *= dim
    if total != len(flat):
        raise ValueError("Flat data length does not match shape")

    def _build(offset: int, dims: Sequence[int]) -> Tuple[Any, int]:
        if not dims:
            return flat[offset], offset + 1
        size = dims[0]
        remainder = dims[1:]
        items = []
        for _ in range(size):
            item, offset = _build(offset, remainder)
            items.append(item)
        return items, offset

    nested, _ = _build(0, shape)
    return nested


def _buffer_pointer(buffer: bytearray) -> int:
    """Return a raw pointer (int) to the start of the buffer."""
    if len(buffer) == 0:
        return 0
    c_buf = (ctypes.c_char * len(buffer)).from_buffer(buffer)
    return ctypes.addressof(c_buf)


def _pack_flat_to_cpu_buffer(flat: Sequence[Any], fmt: str, elem_size: int) -> bytearray:
    buffer = bytearray(len(flat) * elem_size)
    for i, value in enumerate(flat):
        struct.pack_into(fmt, buffer, i * elem_size, value)
    return buffer


def _create_cpu_storage(flat: Sequence[Any], dtype: DType, elem_size: int, device: Device) -> Storage:
    fmt = _STRUCT_FORMATS[dtype]
    buffer = _pack_flat_to_cpu_buffer(flat, fmt, elem_size)
    data_ptr = _buffer_pointer(buffer)
    return Storage(data_ptr=data_ptr, bytes=len(buffer), dtype=dtype, device=device, cpu_buffer=buffer)


def _create_cuda_storage(flat: Sequence[Any], dtype: DType, elem_size: int, device: Device) -> Storage:
    try:
        from .cuda import allocator
    except ImportError as exc:
        raise NotImplementedError("CUDA runtime is not available.") from exc

    if not allocator.is_available():
        raise NotImplementedError("CUDA runtime could not be loaded.")

    num_bytes = len(flat) * elem_size
    data_ptr = allocator.cuda_malloc(num_bytes)
    if num_bytes:
        fmt = _STRUCT_FORMATS[dtype]
        host_buffer = _pack_flat_to_cpu_buffer(flat, fmt, elem_size)
        host_ptr = _buffer_pointer(host_buffer)
        allocator.cuda_memcpy_host_to_device(data_ptr, host_ptr, num_bytes)

    return Storage(
        data_ptr=data_ptr,
        bytes=num_bytes,
        dtype=dtype,
        device=device,
        cpu_buffer=None,
    )


def make_storage(data: Any, device: str, dtype: Optional[DType] = None) -> Tuple[Storage, Tuple[int, ...]]:
    if dtype is None:
        dtype = infer_dtype(data)

    shape = infer_shape(data)
    flat = flatten_data(data)
    device_obj = Device.from_value(device)
    elem_size = DTYPE_SIZES[dtype]

    if device_obj.type == DeviceType.CPU:
        storage = _create_cpu_storage(flat, dtype, elem_size, device_obj)
    elif device_obj.type == DeviceType.CUDA:
        storage = _create_cuda_storage(flat, dtype, elem_size, device_obj)
    else:
        raise ValueError(f"Unsupported device type: {device_obj.type}")

    return storage, shape


def _element_byte_offset(strides: Sequence[int], index: Sequence[int], offset: int, elem_size: int) -> int:
    element_index = offset
    for idx, stride in zip(index, strides):
        element_index += idx * stride
    return element_index * elem_size


def _iter_indices(shape: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    if not shape:
        yield ()
        return
    def _rec(prefix: Tuple[int, ...], dims: Sequence[int]) -> Iterable[Tuple[int, ...]]:
        if not dims:
            yield prefix
            return
        for i in range(dims[0]):
            yield from _rec(prefix + (i,), dims[1:])
    yield from _rec((), shape)


def read_flat(storage: Storage, shape: Sequence[int], strides: Sequence[int], offset: int) -> List[Any]:
    if storage.device.type != DeviceType.CPU:
        raise NotImplementedError("Reading non-CPU storage is not supported yet.")
    if storage.cpu_buffer is None:
        raise ValueError("CPU storage is missing its backing buffer.")

    flat: List[Any] = []
    elem_size = DTYPE_SIZES[storage.dtype]
    fmt = _STRUCT_FORMATS[storage.dtype]
    for idx in _iter_indices(shape):
        byte_offset = _element_byte_offset(strides, idx, offset, elem_size)
        (value,) = struct.unpack_from(fmt, storage.cpu_buffer, byte_offset)
        flat.append(value)
    return flat
