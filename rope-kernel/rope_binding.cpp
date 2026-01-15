#include <torch/extension.h>

// Declaration of the function from the .cu file
void rope_cuda_forward(torch::Tensor x, torch::Tensor freqs);

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rope_cuda_forward, "RoPE Forward (CUDA)");
}