import torch
import os
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
print("Compiling and loading custom CUDA kernel...")
rope_custom = load(
    name='rope_custom', 
    sources=['rope_binding.cpp', 'rope_kernel.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def rope_torch(x, freqs):
    """
    Manual PyTorch implementation of Half-Split RoPE 
    to verify the CUDA kernel.
    """
    B, Nh, S, D = x.shape
    half_d = D // 2
    
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    freqs = freqs.unsqueeze(0).unsqueeze(0) # [1, 1, S, D/2]
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def benchmark_cuda(func, args, name, n_warmup=5, n_runs=10):
    """
    Accurately measures CUDA execution time using CUDA Events.
    """
    # Warmup
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_runs):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event) / n_runs
    print(f"{name:<25} | {elapsed_time_ms:.4f} ms per iter")
    return elapsed_time_ms

def test():
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # correctness check
    print("\n--- Correctness Check ---")
    B_chk, Nh_chk, S_chk, D_chk = 2, 4, 128, 64
    half_d_chk = D_chk // 2
    
    x = torch.randn(B_chk, Nh_chk, S_chk, D_chk, device=device, dtype=torch.float32)
    freqs = torch.randn(S_chk, half_d_chk, device=device, dtype=torch.float32)

    x_manual = x.clone()
    y_ref = rope_torch(x_manual, freqs)
    
    x_cuda = x.clone() 
    rope_custom.forward(x_cuda, freqs)
    
    diff = (y_ref - x_cuda).abs().max().item()
    if torch.allclose(y_ref, x_cuda, atol=1e-5):
        print(f"✅ Success! Max Diff: {diff:.6f}")
    else:
        print(f"❌ Failure! Max Diff: {diff:.6f}")
        return

    # correctness check
    print("\n--- Performance Benchmark ---")
    
    B_bench = 32      
    Nh_bench = 32    
    S_bench = 1024     
    D_bench = 512
        
    print(f"Config: Batch={B_bench}, Heads={Nh_bench}, SeqLen={S_bench}, Dim={D_bench}")
    
    x_bench = torch.randn(B_bench, Nh_bench, S_bench, D_bench, device=device)
    freqs_bench = torch.randn(S_bench, D_bench // 2, device=device)
    
    time_py = benchmark_cuda(
        rope_torch, 
        (x_bench, freqs_bench), 
        "PyTorch Manual"
    )

    time_custom = benchmark_cuda(
        rope_custom.forward, 
        (x_bench, freqs_bench), 
        "Custom CUDA Kernel"
    )

    print(f"-" * 45)
    print(f"Speedup: {time_py / time_custom:.2f}x")

if __name__ == "__main__":
    test()