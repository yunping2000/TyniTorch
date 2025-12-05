from dataclasses import dataclass
from typing import Any

import torch

from .typing import Device

"""
Storage class is the underlying data structure for TyniTorch tensors.

In v0, we simply wrap a torch.Tensor and keep track of the device.
We should decouple from torch in v1, but for now this is sufficient.
"""
@dataclass
class Storage:
    impl: torch.Tensor
    device: Device # "cpu", "cuda", etc.


def make_storage(data: Any, device: str = "cpu") -> Storage:
    device_obj = Device.from_value(device)
    device_str = str(device_obj)

    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data, device=device_str)
    else:
        tensor = data.to(device_str)

    return Storage(impl=tensor, device=device_obj)
