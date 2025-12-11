#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <cuda_runtime.h>

#include "kernels/add.cuh"
#include "kernels/contiguous.cuh"

namespace py = pybind11;

PYBIND11_MODULE(tynitorch_cuda, m) {
    m.def(
        "add_f32",
        [](std::uintptr_t a_ptr,
           std::uintptr_t b_ptr,
           std::uintptr_t out_ptr,
           std::size_t n,
           int device_index) {
            const float* a = reinterpret_cast<const float*>(a_ptr);
            const float* b = reinterpret_cast<const float*>(b_ptr);
            float* out = reinterpret_cast<float*>(out_ptr);
            // This is just a normal function call; kernel launch happens in .cu
            add_f32_launcher(a, b, out, n, device_index);
        },
        "Elementwise add on float32 device buffers"
    );

    m.def(
        "contiguous_f32",
        [](std::uintptr_t src_ptr,
           std::vector<std::size_t> shape, // python list is converted to std::vector
           std::vector<std::size_t> strides,  // python list is converted to std::vector
           std::uintptr_t dst_ptr,
           int device_index) {
            const float* src = reinterpret_cast<const float*>(src_ptr);
            float* dst = reinterpret_cast<float*>(dst_ptr);
            if (shape.size() != strides.size()) {
                throw std::invalid_argument("shape and strides must have the same length");
            }
            std::size_t rank = shape.size();
            std::size_t numel = 1;
            for (std::size_t dim : shape) {
                numel *= dim;
            }

            std::size_t* d_shape = nullptr;
            std::size_t* d_strides = nullptr;
            cudaSetDevice(device_index);
            cudaMalloc(&d_shape, sizeof(std::size_t) * rank);
            cudaMalloc(&d_strides, sizeof(std::size_t) * rank);
            cudaMemcpy(d_shape, shape.data(), sizeof(std::size_t) * rank, cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, strides.data(), sizeof(std::size_t) * rank, cudaMemcpyHostToDevice);

            try {
                contiguous_f32_launcher(src, d_shape, d_strides, rank, dst, numel, device_index);
            } catch (...) {
                cudaFree(d_shape);
                cudaFree(d_strides);
                throw;
            }

            cudaFree(d_shape);
            cudaFree(d_strides);
        },
        "Make a strided float32 tensor contiguous on device"
    );
}
