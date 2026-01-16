#include <torch/extension.h>

// Declaration of the function from the .cu file
torch::Tensor matrix_mul(torch::Tensor A, torch::Tensor B);

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matrix_mul, "Coarsened matrix mul");
}