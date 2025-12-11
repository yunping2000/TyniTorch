from ..dispatcher import get_kernel
from ..tensor import Tensor


def add(t1: Tensor, t2: Tensor) -> Tensor:
    if t1.device != t2.device:
        raise ValueError(f"Device mismatch: {t1.device} vs {t2.device}")

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")

    kernel = get_kernel("add", t1.device)
    return kernel(t1, t2)
