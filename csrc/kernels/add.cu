// csrc/add_kernel.cu
#include "add.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstddef>

template <typename T>
__global__ void add_kernel(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ out,
                           std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    std::size_t stride = blockDim.x * gridDim.x; // total number of threads
    for (std::size_t i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

template <typename T>
static void launch_add(const T* a,
                       const T* b,
                       T* out,
                       std::size_t n,
                       int device_index) {
    cudaSetDevice(device_index);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size; // grid_size = num_blocks

    add_kernel<T><<<grid_size, block_size>>>(a, b, out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Concrete float32 launcher with a simple, unmangled name
void add_f32_launcher(const float* a,
                      const float* b,
                      float* out,
                      std::size_t n,
                      int device_index) {
    launch_add<float>(a, b, out, n, device_index);
}
