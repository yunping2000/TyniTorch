from ...storage import Storage
from ...tensor import Tensor, compute_default_strides
from ...typing import DType, DTYPE_SIZES


def add_gpu(t1: Tensor, t2: Tensor) -> Tensor:
    """GPU-accelerated addition using CUDA kernel for float32."""
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    if t1.dtype != t2.dtype:
        raise ValueError(f"Dtype mismatch: {t1.dtype} vs {t2.dtype}")
    if t1.dtype != DType.FLOAT32:
        raise NotImplementedError("CUDA add currently supports float32 tensors only.")

    try:
        import tynitorch_cuda
    except ImportError as e:
        raise ImportError("CUDA extension module (tynitorch_cuda) not found. Make sure it's built.") from e

    if t1.num_elements() == 0:
        return Tensor(data=[], device=str(t1.device), dtype=t1.dtype)

    elem_size = DTYPE_SIZES[t1.dtype]
    out_storage = Storage.allocate(t1.num_elements(), t1.dtype, t1.device)
    device_index = t1.device.index or 0

    if t1.is_contiguous() and t2.is_contiguous():
        a_ptr = t1.storage.data_ptr + t1.offset * elem_size
        b_ptr = t2.storage.data_ptr + t2.offset * elem_size
        tynitorch_cuda.add_f32(
            a_ptr,
            b_ptr,
            out_storage.data_ptr,
            t1.num_elements(),
            device_index,
        )
    else:
        shape = list(t1.shape)
        strides_a = list(t1.strides)
        strides_b = list(t2.strides)
        a_ptr = t1.storage.data_ptr + t1.offset * elem_size
        b_ptr = t2.storage.data_ptr + t2.offset * elem_size
        tynitorch_cuda.add_strided_f32(
            a_ptr,
            shape,
            strides_a,
            b_ptr,
            strides_b,
            out_storage.data_ptr,
            device_index,
        )

    return Tensor.from_storage(
        storage=out_storage,
        shape=t1.shape,
        strides=compute_default_strides(t1.shape),
        offset=0,
    )
