from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

_SHAPE = tuple[int, ...]


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"


DTYPE_SIZES = {
    DType.FLOAT32: 4,
    DType.FLOAT64: 8,
    DType.INT32: 4,
    DType.INT64: 8,
}


@dataclass(frozen=True)
class Device:
    type: DeviceType
    index: Optional[int] = None

    def __post_init__(self):
        if self.type == DeviceType.CPU and self.index is not None:
            raise ValueError("CPU device cannot have an index")
        if self.index is not None and self.index < 0:
            raise ValueError("Device index must be non-negative")

    def __str__(self) -> str:
        if self.type == DeviceType.CPU:
            return self.type.value
        if self.index is None:
            return self.type.value
        return f"{self.type.value}:{self.index}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Device):
            return (self.type, self.index) == (other.type, other.index)
        if isinstance(other, str):
            return str(self) == other
        return False

    @classmethod
    def from_value(cls, value: Union["Device", str]) -> "Device":
        """Normalize a Device or string like 'cuda:0' into a Device."""
        if isinstance(value, Device):
            return value

        if not isinstance(value, str):
            raise TypeError(f"Device must be a str or Device, got {type(value)}")

        if value.startswith(DeviceType.CUDA.value):
            parts = value.split(":")
            index = int(parts[1]) if len(parts) > 1 and parts[1] != "" else None
            return cls(DeviceType.CUDA, index=index)

        if value == DeviceType.CPU.value:
            return cls(DeviceType.CPU)

        raise ValueError(f"Unrecognized device string: {value}")
