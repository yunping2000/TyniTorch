from enum import Enum
_SHAPE = tuple[int, ...]

class dtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
