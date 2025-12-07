// csrc/add_bindings.cpp
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstddef>

#include "add_kernel.cuh"

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
        "Elementwise add on float32 device buffers");
}
