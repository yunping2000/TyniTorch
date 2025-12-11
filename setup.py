# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import pybind11

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
    return nvcc


class BuildExtensionWithCUDA(build_ext):
    def build_extensions(self):
        if not CUDA_AVAILABLE:
            return super().build_extensions()
        self._customize_compiler_for_nvcc()
        super().build_extensions()

    def _customize_compiler_for_nvcc(self):
        if NVCC_PATH is None:
            raise RuntimeError("CUDA build requested but nvcc was not found.")
        compiler = self.compiler

        # allow .cu sources
        if ".cu" not in compiler.src_extensions:
            compiler.src_extensions.append(".cu")

        default_compiler_so = compiler.compiler_so[:] if isinstance(compiler.compiler_so, list) else compiler.compiler_so
        nvcc = NVCC_PATH

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


NVCC_PATH = locate_nvcc()
CUDA_AVAILABLE = NVCC_PATH is not None

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

ext_modules = []

if CUDA_AVAILABLE:
    cuda_extension = Extension(
        name="tynitorch_cuda",
        sources=[
            "csrc/kernels/add.cu",
            "csrc/kernels/contiguous.cu",
            "csrc/bindings.cpp",
        ],
        include_dirs=[pybind11.get_include()],
        # Link against the CUDA runtime (cudart). We will try to determine
        # the CUDA library directory from CUDA_HOME / CUDA_PATH or nvcc.
        libraries=[],
        library_dirs=[],
        runtime_library_dirs=[],
        language="c++",
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        }, # type: ignore
    )

    try:
        inferred_cuda_home = (
            os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
            or os.path.dirname(os.path.dirname(NVCC_PATH))
        )
        cuda_lib_dir = os.path.join(inferred_cuda_home, "lib64")
        cuda_extension.libraries = list(cuda_extension.libraries) + ["cudart"]
        cuda_extension.library_dirs = list(cuda_extension.library_dirs) + [cuda_lib_dir]
        cuda_extension.runtime_library_dirs = list(cuda_extension.runtime_library_dirs) + [cuda_lib_dir]
    except Exception:
        # If we cannot derive CUDA paths, continue without linker hints; build may still succeed.
        pass

    ext_modules.append(cuda_extension)
else:
    print("CUDA toolkit (nvcc) not found; building CPU-only package.")

setup(
    name="tynitorch",
    version="0.0.0",
    packages=["tynitorch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionWithCUDA} if CUDA_AVAILABLE else {},
)
