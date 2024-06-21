import os, glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "cxx": [
        "-O3",
        "-fdiagnostics-color=always",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
    ]
}

library_name = "llmfs"
this_dir = os.path.dirname(os.path.curdir)
extensions_dir = os.path.join(this_dir, library_name, "csrc")
sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
sources += cuda_sources

def main():
    setup(name="llmfs",
          packages=find_packages(),
          ext_modules=[
              CUDAExtension(
                  name="llmfs._C",
                  sources=["csrc/pybind.cpp", "csrc/layer/linear.cu"],
                  extra_compile_args=extra_compile_args
              )
          ],
          cmdclass={"build_ext": BuildExtension},)
          #install_requires=["torch"])


if __name__ == '__main__':
    main()
