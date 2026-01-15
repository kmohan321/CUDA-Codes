#include <torch/extension.h>

// Declaration of the function from the .cu file
torch::Tensor gated_gelu_forward(torch::Tensor x);

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gated_gelu_forward, "Gated GeLU Forward (CUDA)");
}