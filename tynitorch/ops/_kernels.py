from ..storage import read_flat, reshape_flat
from ..tensor import Tensor
from ..dispatcher import register_kernel


def add_cpu(t1: Tensor, t2: Tensor) -> Tensor:
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    flat1 = read_flat(t1.storage, t1.shape, t1.strides, t1.offset)
    flat2 = read_flat(t2.storage, t2.shape, t2.strides, t2.offset)
    flat_sum = [a + b for a, b in zip(flat1, flat2)]
    nested = reshape_flat(flat_sum, t1.shape)
    return Tensor(
        data=nested,
        device=str(t1.device),
        dtype=t1.dtype,  # assuming both tensors have the same dtype
    )


def add_gpu(t1: Tensor, t2: Tensor) -> Tensor:
    raise NotImplementedError("CUDA backend is not implemented yet.")


register_kernel("add", "cpu", add_cpu)
register_kernel("add", "cuda", add_gpu)
