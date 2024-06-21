#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "layer/linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward_cuda", &linear_forward_cuda, "Linear forward (CUDA)");
}