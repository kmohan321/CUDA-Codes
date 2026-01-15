#include <iostream>
#include <cuda_runtime.h>

__global__ void vec_add(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {

    // We need ~16 Million elements (64MB per array) to stress the GPU memory.
    int N = 1 << 24; // ~16.7 Million elements
    size_t bytes = N * sizeof(float);

    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Total Data Transfer: " << (3.0 * bytes) / (1024*1024*1024) << " GB" << std::endl;

    float *a = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, bytes);
    cudaMalloc((void**)&b_d, bytes);
    cudaMalloc((void**)&c_d, bytes);

    cudaMemcpy(a_d, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //warp up
    vec_add<<<blocks, threads>>>(a_d, b_d, c_d, N);

    cudaEventRecord(start);
    vec_add<<<blocks, threads>>>(a_d, b_d, c_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    double total_bytes = 3.0 * N * sizeof(float);
    double seconds = milliseconds / 1000.0;
    double bandwidth = (total_bytes / seconds) / 1e9; // GB/s

    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c);
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
    
    return 0;
}