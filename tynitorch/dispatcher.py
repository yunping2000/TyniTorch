from typing import Any, Callable, Dict, Tuple

from .typing import Device

Key = Tuple[str, str]  # (op_name, device)
_REGISTRY: Dict[Key, Callable[..., Any]] = {}


def _canonical_device(device: str | Device) -> str:
    """Collapse device identifiers to device type only (cpu/cuda)."""
    return Device.from_value(device).type.value


def register_kernel(op_name: str, device: str | Device, kernel: Callable[..., Any]) -> None:
    """Register a kernel for a specific operation and device."""
    key = (op_name, _canonical_device(device))
    if key in _REGISTRY:
        raise ValueError(f"Kernel already registered for key: {key}")
    _REGISTRY[key] = kernel

def get_kernel(op_name: str, device: str | Device) -> Callable[..., Any]:
    """Retrieve the kernel for a specific operation and device."""
    key = (op_name, _canonical_device(device))
    if key not in _REGISTRY:
        raise KeyError(f"No kernel registered for key: {key}")
    return _REGISTRY[key]
