from dataclasses import dataclass
import torch

"""
Storage class is the underlying data structure for TyniTorch tensors.

In v0, we simply wrap a torch.Tensor and keep track of the device.
We should decouple from torch in v1, but for now this is sufficient.
"""
@dataclass
class Storage:
    impl: torch.Tensor
    device: str # "cpu", "cuda", etc.


def make_storage(data: torch.Tensor, device: str = "cpu") -> Storage:
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data, device=device)
    else:
        tensor = data.to(device)
    return Storage(impl=tensor, device=device)
