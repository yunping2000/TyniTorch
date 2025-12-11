// csrc/add_kernel.cuh
#pragma once

#include <cstddef>

/// Host launcher for float32 add on CUDA.
/// This is a normal C++ function (no __global__, no <<<>>> in the signature).
void add_f32_launcher(const float* a,
                      const float* b,
                      float* out,
                      std::size_t n,
                      int device_index);

/// Host launcher for float32 add on CUDA supporting arbitrary strides.
void add_f32_strided_launcher(const float* a,
                              const float* b,
                              float* out,
                              const std::size_t* shape,
                              const std::size_t* strides_a,
                              const std::size_t* strides_b,
                              std::size_t rank,
                              std::size_t n,
                              int device_index);
