from ..dispatcher import get_kernel
from ..tensor import Tensor

def contiguous(t: Tensor) -> Tensor:
    if t.is_contiguous():
        return t
    
    kernel = get_kernel("contiguous", t.device)
    return kernel(t)
