from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "nvcc": [
        "-O3",
        "-std=c++11",
        "--use-fast-math",
        "--thread=8",
    ]
}


def main():
    setup(name="llmfs",
          packages=find_packages(),
          ext_modules=[
              CUDAExtension(
                  name="llmfs",
                  sources=[
                      "csrc/pybind.cpp",
                      "csrc/test.cu"
                  ],
                  extra_compile_args=extra_compile_args
              )
          ],
          cmdclass={"build_ext": BuildExtension},
          install_requires=["torch"])


if __name__ == '__main__':
    main()
