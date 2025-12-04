from typing import Callable, Dict, Tuple, Any

Key = Tuple[str, str] # (op_name, device)
_REGISTRY: Dict[Key, Callable[..., Any]] = {}


def register_kernel(op_name: str, device: str, kernel: Callable[..., Any]) -> None:
    """Register a kernel for a specific operation and device."""
    key = (op_name, device)
    if key in _REGISTRY:
        raise ValueError(f"Kernel already registered for key: {key}")
    _REGISTRY[key] = kernel

def get_kernel(op_name: str, device: str) -> Callable[..., Any]:
    """Retrieve the kernel for a specific operation and device."""
    key = (op_name, device)
    if key not in _REGISTRY:
        raise KeyError(f"No kernel registered for key: {key}")
    return _REGISTRY[key]