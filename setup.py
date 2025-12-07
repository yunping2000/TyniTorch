# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import pybind11

from setuptools.command.build_ext import build_ext
import os

def find_in_path(executable, path=None):
    if path is None:
        path = os.environ.get("PATH", "")
    for p in path.split(os.pathsep):
        candidate = os.path.join(p, executable)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

def locate_nvcc():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(nvcc) and os.access(nvcc, os.X_OK):
            return nvcc
    nvcc = find_in_path("nvcc")
    if nvcc is None:
        raise RuntimeError("Could not find 'nvcc'. Set CUDA_HOME or add nvcc to PATH.")
    return nvcc


class BuildExtensionWithCUDA(build_ext):
    def build_extensions(self):
        self._customize_compiler_for_nvcc()
        super().build_extensions()

    def _customize_compiler_for_nvcc(self):
        compiler = self.compiler

        # allow .cu sources
        if ".cu" not in compiler.src_extensions:
            compiler.src_extensions.append(".cu")

        default_compiler_so = compiler.compiler_so[:] if isinstance(compiler.compiler_so, list) else compiler.compiler_so
        nvcc = locate_nvcc()

        super_compile = compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # normalize extra_postargs into two lists: for cxx and nvcc
            if isinstance(extra_postargs, dict):
                postargs_cxx = extra_postargs.get("cxx", []) or []
                postargs_nvcc = extra_postargs.get("nvcc", []) or []
            elif extra_postargs is None:
                postargs_cxx = []
                postargs_nvcc = []
            else:
                # treat as generic list for both
                postargs_cxx = extra_postargs
                postargs_nvcc = extra_postargs

            if src.endswith(".cu"):
                # use nvcc
                compiler.set_executable("compiler_so", nvcc)
                try:
                    # pass nvcc args as extra_postargs (must be a list, not None)
                    super_compile(obj, src, ext, cc_args, postargs_nvcc, pp_opts)
                finally:
                    # restore default compiler
                    compiler.compiler_so = default_compiler_so
            else:
                # normal C++ file
                super_compile(obj, src, ext, cc_args, postargs_cxx, pp_opts)

        compiler._compile = _compile


# Common compile flags
cxx_args = ["-O3", "-std=c++17"]

# For nvcc:
# -std=c++17 for host, --compiler-options to pass flags to host compiler
nvcc_args = [
    "-O3",
    "-std=c++17",
    "-allow-unsupported-compiler",
    "--compiler-options", "-fPIC",
]

ext_modules = [
    Extension(
        name="tynitorch_cuda",
        sources=[
            "csrc/add_kernel.cu",
            "csrc/add_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        }, # type: ignore
    )
]

setup(
    name="tynitorch",
    version="0.0.0",
    packages=["tynitorch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionWithCUDA},
)
