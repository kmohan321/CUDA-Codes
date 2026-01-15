#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "=======================================" << std::endl;
        
        // 1. Architecture & Core Count
        std::cout << "  Compute Capability:       " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors (SMs):    " << prop.multiProcessorCount << std::endl;
        
        // 2. The "Resident" Limits (Crucial for Occupancy)
        std::cout << "  Max Threads per SM:       " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Threads per Block:    " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp Size:                " << prop.warpSize << std::endl;

        // 3. Memory Hierarchies
        std::cout << "  Global Memory (VRAM):     " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  L2 Cache Size:            " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        
        // 4. Local Resource Limits (The "Tetris" constraints)
        std::cout << "  Total Registers per SM:   " << prop.regsPerMultiprocessor << std::endl; // The hard limit
        std::cout << "  Max Registers per Block:  " << prop.regsPerBlock << std::endl;
        std::cout << "  Shared Mem per SM:        " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
        std::cout << "  Shared Mem per Block:     " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        
        std::cout << std::endl;
    }

    return 0;
}