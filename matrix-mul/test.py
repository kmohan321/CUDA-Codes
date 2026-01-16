import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

print("Compiling CUDA kernel... ")
matmul_custom = load(name='matmul_extension', sources=['mm_binding.cpp', 'mm_v3.cu'], extra_cuda_cflags=["-O3"])
print("Compilation complete.\n")

def pytorch_matmul(a, b):
    """
    Computes Matrix Multiplication C = A @ B
    """
    return torch.matmul(a, b)

# Correctness Check
def check_correctness():
    print("--- Checking Correctness ---")
    # Define problem size (M, N, K)
    M, N, K = 128, 128, 128
    
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    ref_out = pytorch_matmul(a, b)
    cuda_out = matmul_custom.forward(a, b)
    
    if torch.allclose(ref_out, cuda_out, atol=1e-3, rtol=1e-3):
        print("✅ Correctness Pass: Custom kernel matches PyTorch output.")
    else:
        print("❌ Correctness Fail!")
        diff = (ref_out - cuda_out).abs().max()
        print(f"Max difference: {diff.item()}")

# Benchmarking 
def benchmark_kernel(func, name, n_warmup=10, n_runs=10):
    # Warmup
    for _ in range(n_warmup):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_runs):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / n_runs
    
    print(f"{name:<20}: {avg_time_ms:.4f} ms")
    return avg_time_ms

def run_benchmark():
    print("\n--- Benchmarking ---")
    
    M = 4096
    N = 4096
    K = 4096
    
    print(f"Config: M={M}, N={N}, K={K}")
    
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    bench_ref = lambda: pytorch_matmul(a, b)
    t_ref = benchmark_kernel(bench_ref, "PyTorch (cuBLAS)")
    glops_ref = (2 * M * N * K )/ (t_ref * 1e6) 
    
    bench_cuda = lambda: matmul_custom.forward(a, b)
    t_cuda = benchmark_kernel(bench_cuda, "Custom CUDA")
    glops_cuda = (2 * M * N * K )/ (t_cuda * 1e6) 

    print("-" * 40)
    print(f"Speedup: {t_ref / t_cuda:.2f}x")
    
    print(f"Cublas GFLOPs: {glops_ref:.3f} ")
    print(f"Custom CUDA GFLOPs: {glops_cuda:.3f} ")

if __name__ == "__main__":
    check_correctness()
    run_benchmark()