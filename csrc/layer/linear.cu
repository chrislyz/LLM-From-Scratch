#include <torch/extension.h>
#include <cuda_runtime.h>

#include "linear.h"

__global__ void LinearForwardKernel() {
}

void linear_forward_cuda(int in_features, int out_features, bool bias) {

}

void linear_backward_cuda() {
}