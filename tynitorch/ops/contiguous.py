from ..dispatcher import get_kernel
from ..tensor import Tensor

def contiguous(t: Tensor) -> Tensor:
    kernel = get_kernel("contiguous", t.device)
    return kernel(t)
