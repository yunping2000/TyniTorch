from ...storage import Storage
from ...tensor import Tensor

def contiguous_cpu(t: Tensor) -> Tensor:
    # Follow t.strides to create a contiguous copy
    flat_data = t.storage.read_flat(t.shape, t.strides, t.offset)

    storage = Storage.allocate(t.num_elements(), t.dtype, t.device)
    storage.copy_from_iterable(flat_data)

    return Tensor.from_storage(
        storage=storage,
        shape=t.shape,
    )
