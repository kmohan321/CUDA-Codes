## CUDA KERNELS

*Benchmarked on NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)*

### 1. Matrix Multiplication (SGEMM)
Comparison of custom kernels against `cuBLAS` (PyTorch default) for size `4096 x 4096`. Achieved **66% of cuBLAS performance** using thread coarsening.

| Implementation | Performance (GFLOPS) | Speedup vs Naive | Note |
| :--- | :--- | :--- | :--- |
| Naive Implementation | ~750 | 1.0x | Baseline |
| Tiled | ~900 | 1.2x | Reduced Global Mem Access by using shared memory |
| Thread Coarsened (4x) | ~2,950 | 3.9x | Increased Instruction-Level Parallelism (mupliple ops per thread) |
| **Thread Coarsened (8x)** | **~5,000** | **6.6x** | **66% of cuBLAS Speed** |
| *cuBLAS (Reference)* | *~7,500* | *--* | *--* |

### 2. Rotary Positional Embeddings (RoPE)
Implemented as a fused kernel to handle complex number rotations efficiently. The custom kernel parallelize across batch, heads, sequence and head dim and due to the fusion of all operations of rope, this fused kernel drastically reduces memory bandwidth overhead compared to the manual PyTorch implementation.

| Configuration (B, H, S, D) | PyTorch Manual Time | Custom CUDA Time | **Speedup** |
| :--- | :--- | :--- | :--- |
| B=32, H=32, S=1024, D=128 | 27.82 ms | **5.97 ms** | **4.66x** |
| B=32, H=32, S=1024, D=256 | 55.43 ms | **11.93 ms** | **4.65x** |
| B=32, H=32, S=2048, D=128 | 55.51 ms | **11.92 ms** | **4.66x** |
| B=32, H=32, S=2048, D=256 | 3689.77 ms | **23.95 ms** | **154.09x** |

### 3. Gated GELU Activation
A fused element-wise kernel combining the Gated Linear Unit and GELU activation. By fusing the operations, we avoid the kernel launch overhead and intermediate memory writes associated with PyTorch's eager execution of separate kernels of (`element mul`, `gelu`, `chunk`).

| Configuration (Batch, Tokens, Hidden Dim) | PyTorch Eager Time | Custom CUDA Time | **Speedup** |
| :--- | :--- | :--- | :--- |
| B=32, T=1024, Dim=2560 | 6.76 ms | **2.77 ms** | **2.44x** |
| B=32, T=1024, Dim=5120 | 13.57 ms | **5.65 ms** | **2.40x** |
| B=32, T=2048, Dim=2560 | 13.58 ms | **5.54 ms** | **2.45x** |
| B=32, T=2048, Dim=5120 | 27.14 ms | **11.02 ms** | **2.46x** |

### 4. Flash Attention (v1 & v2)
Implementation of exact attention with **O(N)** memory complexity using **Online Softmax** to avoid materializing the full $N \times N$ attention matrix.

#### A. Flash Attention v1
* **Architecture:** Parallelizes computation across **Batch Size ($B$)** and **Number of Heads ($N_h$)**.
* **Loop Structure:** Implements an **outer loop** over Key ($K$) and Value ($V$) blocks, and an **inner loop** over Query ($Q$) blocks.
* **Memory Efficiency:** Reduces HBM reads by keeping tiles in SRAM during computation.

| Configuration (B, Nh, S, D) | Tile Size ($B_r, B_c$) | Execution Time |
| :--- | :--- | :--- |
| B=16, Nh=8, S=1024, D=64 | 16, 16 | **149.30 ms** |

#### B. Flash Attention v2
* **Architecture:** improved parallelism by distributing work across **Batch ($B$)**, **Heads ($N_h$)**, AND **Sequence Length ($S$)** to increase GPU occupancy.
* **Loop Structure:** Optimized to loop **only over Key ($K$) and Value ($V$) blocks** (while parallelizing the $Q$ dimension), reducing loop overhead and redundant operations.

| Configuration (B, Nh, S, D) | Tile Size ($B_r, B_c$) | Execution Time |
| :--- | :--- | :--- |
| B=16, Nh=8, S=1024, D=64 | 16, 8 | **59.51 ms** |