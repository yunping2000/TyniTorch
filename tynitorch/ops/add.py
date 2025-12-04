from ..dispatcher import get_kernel
from ..tensor import Tensor
from . import _kernels  # noqa: F401

"""
The entry point for addition operation.
This function dispatches the addition operation to the appropriate kernel
"""
def add(t1: Tensor, t2: Tensor) -> Tensor:
    if t1.device != t2.device:
        raise ValueError(f"Device mismatch: {t1.device} vs {t2.device}")

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")

    kernel = get_kernel("add", t1.device)

    # Now the kernel returns Tensor directly, so we can just return its result
    # However, we might need to make kernel work in lower levels in the future.
    # In that case, we would need to wrap the result into Tensor here.
    return kernel(t1, t2)

