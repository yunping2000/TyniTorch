import ctypes
import ctypes.util
from typing import Optional


class CudaError(RuntimeError):
    """Raised when a CUDA runtime API call fails."""


_CUDART: Optional[ctypes.CDLL] = None


def _candidate_names() -> list[str]:
    # Common names across Linux, macOS, and Windows
    return [
        "cudart",
        "libcudart.so",
        "libcudart.dylib",
        "cudart64_110.dll",
        "cudart64_120.dll",
    ]


def load_cudart() -> ctypes.CDLL:
    """Load and cache the CUDA runtime library."""
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    last_error: Optional[Exception] = None
    for name in _candidate_names():
        try_name = ctypes.util.find_library(name) or name
        try:
            _CUDART = ctypes.CDLL(try_name)
            break
        except OSError as exc:
            last_error = exc

    if _CUDART is None:
        raise ImportError(
            "Could not load CUDA runtime library; "
            "ensure CUDA is installed and on your loader path."
        ) from last_error

    # Configure signatures once after load
    _CUDART.cudaMalloc.restype = ctypes.c_int
    _CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

    _CUDART.cudaFree.restype = ctypes.c_int
    _CUDART.cudaFree.argtypes = [ctypes.c_void_p]

    _CUDART.cudaGetErrorString.restype = ctypes.c_char_p
    _CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]

    _CUDART.cudaMemcpy.restype = ctypes.c_int
    _CUDART.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    return _CUDART


def is_available() -> bool:
    """Return True if the CUDA runtime can be loaded."""
    try:
        load_cudart()
        return True
    except Exception:
        return False


def _check_error(code: int, fn_name: str) -> None:
    if code == 0:
        return
    cudart = load_cudart()
    msg = cudart.cudaGetErrorString(code)
    human = msg.decode("utf-8") if msg else f"error code {code}"
    raise CudaError(f"{fn_name} failed: {human}")


def cuda_malloc(num_bytes: int) -> int:
    """Allocate device memory and return the raw pointer value."""
    cudart = load_cudart()
    ptr = ctypes.c_void_p()
    code = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(num_bytes))
    _check_error(code, "cudaMalloc")
    return int(ptr.value) if ptr.value is not None else 0


def cuda_free(ptr: int) -> None:
    """Free device memory allocated with cuda_malloc."""
    cudart = load_cudart()
    code = cudart.cudaFree(ctypes.c_void_p(ptr))
    _check_error(code, "cudaFree")


_CUDA_MEMCPY_HOST_TO_DEVICE = 1


def cuda_memcpy(dst_ptr: int, src_ptr: int, num_bytes: int, kind: int) -> None:
    """Copy memory using cudaMemcpy."""
    cudart = load_cudart()
    code = cudart.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(num_bytes),
        ctypes.c_int(kind),
    )
    _check_error(code, "cudaMemcpy")


def cuda_memcpy_host_to_device(dst_ptr: int, src_ptr: int, num_bytes: int) -> None:
    cuda_memcpy(dst_ptr, src_ptr, num_bytes, _CUDA_MEMCPY_HOST_TO_DEVICE)
