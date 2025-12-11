// csrc/kernels/contiguous.cuh
#pragma once

#include <cstddef>

/// Make a strided float32 tensor contiguous on device.
/// `shape` and `strides` are arrays on device of length `rank`.
/// `numel` is total elements to copy.
void contiguous_f32_launcher(const float* src,
                             const std::size_t* shape,
                             const std::size_t* strides,
                             std::size_t rank,
                             float* dst,
                             std::size_t numel,
                             int device_index);
