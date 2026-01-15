#include <torch/extension.h>

torch::Tensor  fa_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa_forward", &fa_forward, "fa_forward");
}
