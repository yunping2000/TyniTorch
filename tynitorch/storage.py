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
    # Raw pointer to the allocated memory.
    data_ptr: int
    num_bytes: int
    dtype: DType
    device: Device
    ref_count: int = 1
    # Keep a reference to the CPU-side buffer so it's not freed by GC.
    _cpu_buffer: Optional[bytearray] = None

    def free(self) -> None:
        """Free Storage resources. CPU memory is auto-freed; CUDA memory must be explicitly freed."""
        if self.device.type == DeviceType.CPU:
            # CPU storage is automatically freed after GC
            return

        # Free CUDA memory
        try:
            from .cuda import allocator
            if self.data_ptr and self.num_bytes:
                allocator.cuda_free(self.data_ptr)
        except ImportError:
            pass

    @classmethod
    def allocate(cls, length: int, dtype: DType, device: str | Device) -> "Storage":
        """Allocate a Storage object for given length/dtype/device."""
        num_bytes = length * DTYPE_SIZES[dtype]
        device_obj = Device.from_value(device)
        host_buf: Optional[bytearray] = None
        if device_obj.type == DeviceType.CPU:
            # Allocate a host bytearray and keep a reference to it to avoid
            # its memory being reclaimed by the GC (which would invalidate
            # the raw pointer returned by _get_buffer_pointer).
            host_buf = bytearray(num_bytes)
            buffer = _get_buffer_pointer(host_buf)
        elif device_obj.type == DeviceType.CUDA:
            buffer = _allocate_buffer_cuda(num_bytes)
        else:
            raise ValueError(f"Unsupported device type: {device_obj.type}")

        storage = cls(
            data_ptr=buffer,
            num_bytes=num_bytes,
            dtype=dtype,
            device=device_obj,
        )
        if device_obj.type == DeviceType.CPU:
            storage._cpu_buffer = host_buf
        return storage

    def copy_from_list(self, data: List[Any]) -> None:
        """Copy a flat sequence of Python values into this storage."""
        flat = flatten(data)

        # Check if flat size <= storage capacity
        elem_size = DTYPE_SIZES[self.dtype]
        if len(flat) * elem_size > self.num_bytes:
            raise ValueError("Flat data exceeds storage capacity")

        if self.device.type == DeviceType.CPU:
            fmt = _STRUCT_FORMATS[self.dtype]
            elem_size = DTYPE_SIZES[self.dtype]
            # If we have a retained CPU bytearray, write into it directly
            if self._cpu_buffer is not None:
                for i, value in enumerate(flat):
                    struct.pack_into(
                        fmt,
                        self._cpu_buffer,
                        i * elem_size,
                        value,
                    )
            else:
                # Fallback: write into raw address using ctypes.memmove
                # TODO: Probably not needed
                for i, value in enumerate(flat):
                    offset_bytes = i * elem_size
                    packed = struct.pack(fmt, value)
                    ctypes.memmove(self.data_ptr + offset_bytes, packed, elem_size)
        elif self.device.type == DeviceType.CUDA:
            try:
                from .cuda import allocator
            except ImportError as exc:
                raise NotImplementedError("CUDA runtime is not available.") from exc

            host_buffer = bytearray(self.num_bytes)
            for i, value in enumerate(flat):
                struct.pack_into(
                    _STRUCT_FORMATS[self.dtype],
                    host_buffer,
                    i * DTYPE_SIZES[self.dtype],
                    value
                )
            if self.num_bytes:
                host_ptr = _get_buffer_pointer(host_buffer)
                allocator.cuda_memcpy_host_to_device(self.data_ptr, host_ptr, self.num_bytes)
        else:
            raise NotImplementedError("Copying to storage on this device is not supported.")

    def read_flat(self, shape: Sequence[int], strides: Sequence[int], offset: int) -> List[Any]:
        """Read values from storage into a flat Python list according to shape/strides/offset."""
        if self.device.type == DeviceType.CPU:
            buffer = bytearray(
                (ctypes.c_char * self.num_bytes).from_address(self.data_ptr)
            )
        elif self.device.type == DeviceType.CUDA:
            try:
                from .cuda import allocator
            except ImportError as exc:
                raise NotImplementedError("CUDA runtime is not available.") from exc
            host_buffer = bytearray(self.num_bytes)
            if self.num_bytes:
                host_ptr = _get_buffer_pointer(host_buffer)
                allocator.cuda_memcpy_device_to_host(host_ptr, self.data_ptr, self.num_bytes)
            buffer = host_buffer
        else:
            raise NotImplementedError("Reading storage on this device is not supported.")

        flat: List[Any] = []
        elem_size = DTYPE_SIZES[self.dtype]
        fmt = _STRUCT_FORMATS[self.dtype]
        for idx in _iter_indices(shape):
            byte_offset = _element_byte_offset(strides, idx, offset, elem_size)
            (value,) = struct.unpack_from(fmt, buffer, byte_offset)
            flat.append(value)
        return flat

def flatten(nested: Any) -> List[Any]:
    """Flatten nested lists/tuples into a flat list."""
    flat: List[Any] = []
    def _walk(x: Any) -> None:
        if isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)
        else:
            flat.append(x)
    _walk(nested)
    return flat


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


def _get_buffer_pointer(buffer: bytearray) -> int:
    """Return a raw pointer (int) to the start of the buffer."""
    if len(buffer) == 0:
        return 0
    c_buf = (ctypes.c_char * len(buffer)).from_buffer(buffer)
    return ctypes.addressof(c_buf)


def _allocate_buffer_cpu(num_bytes: int) -> int: # Returns a raw pointer
    return _get_buffer_pointer(bytearray(num_bytes))


def _allocate_buffer_cuda(num_bytes: int) -> int: # Returns a raw pointer
    try:
        from .cuda import allocator
    except ImportError as exc:
        raise NotImplementedError("CUDA runtime is not available.") from exc

    if not allocator.is_available():
        raise NotImplementedError("CUDA runtime could not be loaded.")    

    return allocator.cuda_malloc(num_bytes)



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


# Note: allocate_storage, copy_flat_to_storage, and read_flat are now
# implemented as `Storage.allocate`, `Storage.copy_from_flat`, and
# `Storage.read_flat` respectively.
