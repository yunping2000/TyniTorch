// csrc/kernels/contiguous.cu
#include "contiguous.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

__global__ void contiguous_kernel_f32(const float* __restrict__ src,
                                      const std::size_t* __restrict__ shape,
                                      const std::size_t* __restrict__ strides,
                                      std::size_t rank,
                                      float* __restrict__ dst,
                                      std::size_t numel) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    std::size_t offset = 0;
    std::size_t remaining = idx;
    // Convert linear index to multi-dimensional index using shape
    for (int dim = static_cast<int>(rank) - 1; dim >= 0; --dim) {
        std::size_t dim_size = shape[dim];
        std::size_t coord = remaining % dim_size;
        remaining /= dim_size;
        offset += coord * strides[dim];
    }
    dst[idx] = src[offset];
}

void contiguous_f32_launcher(const float* src,
                             const std::size_t* shape,
                             const std::size_t* strides,
                             std::size_t rank,
                             float* dst,
                             std::size_t numel,
                             int device_index) {
    if (numel == 0) {
        return;
    }

    cudaSetDevice(device_index);

    const int block_size = 256;
    const int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    contiguous_kernel_f32<<<grid_size, block_size>>>(src, shape, strides, rank, dst, numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
