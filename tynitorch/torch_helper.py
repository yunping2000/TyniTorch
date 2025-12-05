import torch
from .typing import DType


def dtype_torch_2_tyni(numpy_array: torch.Tensor) -> DType:
    """Infer the dtype enum from a torch tensor (previously numpy array).

    Note: function name kept for compatibility but now accepts a `torch.Tensor`.
    """
    t_dtype = numpy_array.dtype

    if t_dtype == torch.float32:
        return DType.FLOAT32
    elif t_dtype == torch.float64:
        return DType.FLOAT64
    elif t_dtype == torch.int32:
        return DType.INT32
    elif t_dtype == torch.int64:
        return DType.INT64
    else:
        raise ValueError(f"Unsupported tensor dtype: {t_dtype}")


def dtype_tyni_2_torch(tyni_dtype: DType) -> torch.dtype:
    """Convert the dtype enum to a torch dtype.

    Note: function name kept for compatibility but now returns a `torch.dtype`.
    """
    if tyni_dtype == DType.FLOAT32:
        return torch.float32
    elif tyni_dtype == DType.FLOAT64:
        return torch.float64
    elif tyni_dtype == DType.INT32:
        return torch.int32
    elif tyni_dtype == DType.INT64:
        return torch.int64
    else:
        raise ValueError(f"Unsupported TyniTorch dtype: {tyni_dtype}")
