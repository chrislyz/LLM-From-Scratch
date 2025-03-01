#include <torch/extension.h>

void my_linear_forward_cuda(int in_features, int out_features, bool bias);