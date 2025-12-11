from ...storage import Storage
from ...tensor import Tensor, compute_default_strides
from ...typing import DType, DTYPE_SIZES


def contiguous_gpu(t: Tensor) -> Tensor:
    if t.dtype != DType.FLOAT32:
        raise NotImplementedError("CUDA contiguous kernel currently supports float32 only.")

    try:
        import tynitorch_cuda
    except ImportError as exc:
        raise RuntimeError("CUDA extension module not available for contiguous") from exc

    elem_size = DTYPE_SIZES[t.dtype]
    src_ptr = t.storage.data_ptr + t.offset * elem_size
    device_index = t.device.index or 0

    storage = Storage.allocate(t.num_elements(), t.dtype, t.device)

    tynitorch_cuda.contiguous_f32(
        src_ptr,
        list(t.shape),
        list(t.strides),
        storage.data_ptr,
        device_index,
    )

    return Tensor.from_storage(
        storage=storage,
        shape=t.shape,
        strides=compute_default_strides(t.shape),
        offset=0,
    )
