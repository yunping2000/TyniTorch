from ..tensor import Tensor
from ..dispatcher import register_kernel, get_kernel

def add_cpu(t1: Tensor, t2: Tensor) -> Tensor:
    result_data = t1.storage.impl + t2.storage.impl
    return Tensor(
        data=result_data,
        device="cpu",
        dtype=t1.dtype, # assuming both tensors have the same dtype
    )

def add_gpu(t1: Tensor, t2: Tensor) -> Tensor:
    if t1.device != t2.device:
        raise ValueError(f"Device mismatch: {t1.device} vs {t2.device}")

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")

    result_data = t1.storage.impl + t2.storage.impl
    return Tensor(
        data=result_data,
        device="cuda",
        dtype=t1.dtype, # assuming both tensors have the same dtype
    )

register_kernel("add", "cpu", add_cpu)
register_kernel("add", "cuda", add_gpu)
