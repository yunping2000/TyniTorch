from ...storage import Storage
from ...tensor import Tensor


def add_gpu(t1: Tensor, t2: Tensor) -> Tensor:
    """GPU-accelerated addition using CUDA kernel for float32."""
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    if t1.dtype != t2.dtype:
        raise ValueError(f"Dtype mismatch: {t1.dtype} vs {t2.dtype}")

    try:
        import tynitorch_cuda
    except ImportError as e:
        raise ImportError("CUDA extension module (tynitorch_cuda) not found. Make sure it's built.") from e

    if t1.num_elements() == 0:
        # Empty tensor
        return Tensor(data=[], device=str(t1.device), dtype=t1.dtype)

    out_storage = Storage.allocate(t1.num_elements(), t1.dtype, t1.device)

    tynitorch_cuda.add_f32(
        t1.storage.data_ptr,
        t2.storage.data_ptr,
        out_storage.data_ptr,
        t1.num_elements(),
        0,  # device_index (assume device 0 for now)
    )

    return Tensor.from_storage(
        storage=out_storage,
        shape=t1.shape,
        strides=t1.strides,
        offset=0,
    )
