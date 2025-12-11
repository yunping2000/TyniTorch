from ...storage import reshape_flat
from ...tensor import Tensor


def add_cpu(t1: Tensor, t2: Tensor) -> Tensor:
    """CPU implementation of tensor addition."""
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    if t1.dtype != t2.dtype:
        raise ValueError(f"Dtype mismatch: {t1.dtype} vs {t2.dtype}")
    flat1 = t1.storage.read_flat(t1.shape, t1.strides, t1.offset)
    flat2 = t2.storage.read_flat(t2.shape, t2.strides, t2.offset)
    flat_sum = [a + b for a, b in zip(flat1, flat2)]
    nested = reshape_flat(flat_sum, t1.shape)
    return Tensor(
        data=nested,
        device=str(t1.device),
        dtype=t1.dtype,  # assuming both tensors have the same dtype
    )


__all__ = ["add_cpu"]
