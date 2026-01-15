import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

print("Compiling CUDA kernel... ")
gated_gelu = load(name='gated_gelu_extension', 
                  sources=['kernel_binding.cpp', 'gelu_kernel.cu'], extra_cuda_cflags=["-O3"])
print("Compilation complete.\n")

def pytorch_gated_gelu(x):
    """
    Computes GELU(h) * (g + 1)
    Input x: [..., 2*d]
    Split into h, g along last dim
    """
    h, g = x.chunk(2, dim=-1)
    return F.gelu(h, approximate='none') * (g + 1.0)

# Correctness Check
def check_correctness():
    print("--- Checking Correctness ---")
    B, S, D = 16, 128, 4096 # D is full width (2*d)
    
    x = torch.randn(B, S, D, device='cuda', dtype=torch.float32)
    
    ref_out = pytorch_gated_gelu(x)
    cuda_out = gated_gelu.forward(x)
    cuda_out = cuda_out.view(B, S, -1)
    
    if torch.allclose(ref_out, cuda_out, atol=1e-5):
        print("✅ Correctness Pass: Custom kernel matches PyTorch output.")
    else:
        print("❌ Correctness Fail!")
        diff = (ref_out - cuda_out).abs().max()
        print(f"Max difference: {diff.item()}")


# Benchmarking with CUDA Events
def benchmark_kernel(func, name, x, n_warmup=10, n_runs=100):
    # Warmup
    for _ in range(n_warmup):
        func(x)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_runs):
        func(x)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (elapsed_time_ms / n_runs) * 1000
    
    print(f"{name:<20}: {avg_time_us:.2f} us")
    return avg_time_us

def run_benchmark():
    print("\n--- Benchmarking ---")
    
    batch = 32
    tokens = 2048
    hidden_dim = 2560 
    
    print(f"Config: Batch={batch}, Tokens={tokens}, Hidden_Dim (2d)={hidden_dim}")
    
    x = torch.randn(batch, tokens, hidden_dim, device='cuda', dtype=torch.float32)
    
    bench_ref = lambda t: pytorch_gated_gelu(t)
    t_ref = benchmark_kernel(bench_ref, "PyTorch (Eager)", x)
    
    bench_cuda = lambda t: gated_gelu.forward(t)
    t_cuda = benchmark_kernel(bench_cuda, "Custom CUDA", x)

    print("-" * 40)
    print(f"Speedup: {t_ref / t_cuda:.2f}x")

if __name__ == "__main__":
    check_correctness()
    run_benchmark()