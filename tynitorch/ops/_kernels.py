from ..storage import reshape_flat, Storage
from ..tensor import Tensor
from ..dispatcher import register_kernel
from ..typing import DType


def _compute_strides(shape):
    """Compute row-major strides in units of elements."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def add_cpu(t1: Tensor, t2: Tensor) -> Tensor:
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    flat1 = t1.storage.read_flat(t1.shape, t1.strides, t1.offset)
    flat2 = t2.storage.read_flat(t2.shape, t2.strides, t2.offset)
    flat_sum = [a + b for a, b in zip(flat1, flat2)]
    nested = reshape_flat(flat_sum, t1.shape)
    return Tensor(
        data=nested,
        device=str(t1.device),
        dtype=t1.dtype,  # assuming both tensors have the same dtype
    )


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

    # Get total number of elements
    total_elements = 1
    for dim in t1.shape:
        total_elements *= dim
    
    if total_elements == 0:
        # Empty tensor
        return Tensor(data=[], device=str(t1.device), dtype=t1.dtype)

    # Call CUDA kernel
    # Create output storage on GPU
    out_storage = Storage.allocate(total_elements, t1.dtype, t1.device)

    tynitorch_cuda.add_f32(
        t1.storage.data_ptr,
        t2.storage.data_ptr,
        out_storage.data_ptr,
        total_elements,
        0  # device_index (assume device 0 for now)
    )

    # Return new Tensor wrapping the output storage
    return Tensor.from_storage(
        storage=out_storage,
        shape=t1.shape,
        strides=t1.strides,
        offset=0,
    )


register_kernel("add", "cpu", add_cpu)
register_kernel("add", "cuda", add_gpu)
