#include <torch/extension.h>

// Declaration of the function from the .cu file
torch::Tensor layer_norm_forward(torch::Tensor x, float eps);

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layer_norm_forward, "Layer Norm Kernel");
}