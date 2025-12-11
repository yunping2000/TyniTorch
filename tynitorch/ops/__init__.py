from ..dispatcher import register_kernel
from .cpu import add_cpu, contiguous_cpu
from .cuda import add_gpu
from .add import add
from .contiguous import contiguous

# Register available kernels when ops is imported.
register_kernel("add", "cpu", add_cpu)
register_kernel("add", "cuda", add_gpu)

register_kernel("contiguous", "cpu", contiguous_cpu)

__all__ = ["add", "contiguous"]
