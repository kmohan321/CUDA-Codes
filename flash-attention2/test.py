import torch
import os 
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
fa_custom = load(name='fa_custom_2', sources=['fa.cpp', 'fa2_kernel.cu'], extra_cuda_cflags=['-O3'])

def reference_attention(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.transpose(-1, -2)
    S = S / torch.sqrt(torch.tensor(d))
    P = torch.softmax(S, dim=-1)
    return P @ V

#parameters
TOL = 1e-4
device = "cuda"
dtype = torch.float32

Br = 16
Bc = 8
B = 16
Nh = 8
T  = 1024
d  = 64

torch.manual_seed(0)

Q = torch.randn(B, Nh, T, d, device=device, dtype=dtype)
K = torch.randn(B, Nh, T, d, device=device, dtype=dtype)
V = torch.randn(B, Nh, T, d, device=device, dtype=dtype)
O_flash = None
O_ref   = None


# -------------------------------
# Correctness check
# -------------------------------
print("Running correctness check...")

O_flash = fa_custom.fa_forward(Q, K, V)

O_ref = reference_attention(Q, K, V)

max_error = (O_flash - O_ref).abs().max().item()
mean_error = (O_flash - O_ref).abs().mean().item()

print(f"Max error  : {max_error:e}")
print(f"Mean error : {mean_error:e}")


if max_error > TOL:
    print("❌ Incorrect result. Skipping timing.")
    exit(1)

print("✅ Correctness PASSED")

# -------------------------------
# Timing utilities
# -------------------------------
def time_cuda(fn, iters=10):
    
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    return elapsed_ms / iters


# -------------------------------
# Timing FlashAttention
# -------------------------------
flash_time_ms = time_cuda(
    lambda: fa_custom.fa_forward(Q, K, V)
)

# -------------------------------
# Timing PyTorch reference
# -------------------------------
torch_time_ms = time_cuda(
    lambda: reference_attention(Q, K, V)
)

print("\nTiming results (ms):")
print(f"B: {B}, Nh: {Nh}, s: {T}, d: {d}, Br: {Br}, Bc: {Bc}")
print(f"FlashAttention CUDA : {flash_time_ms:.3f} ms")
print(f"PyTorch reference   : {torch_time_ms:.3f} ms")
print(f"Speedup             : {torch_time_ms / flash_time_ms:.2f}×")
