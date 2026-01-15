#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void check_cuda(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

void check_cublas(cublasStatus_t result, const char *func) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error in " << func << std::endl;
        exit(1);
    }
}

int main() {
    int N = 4096;
    size_t bytes = N * N * sizeof(float);

    std::cout << "Benchmarking cuBLAS SGEMM for N=" << N << std::endl;

    // Allocate Host Memory
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    // Allocate Device Memory
    float *dA, *dB, *dC;
    check_cuda(cudaMalloc(&dA, bytes), "cudaMalloc A");
    check_cuda(cudaMalloc(&dB, bytes), "cudaMalloc B");
    check_cuda(cudaMalloc(&dC, bytes), "cudaMalloc C");

    check_cuda(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice), "Memcpy A");
    check_cuda(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice), "Memcpy B");

    // --- cuBLAS Setup ---
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    float alpha = 1.0f;
    float beta = 0.0f;

    // Events for accurate timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    // Note: We swap A and B to handle Row-Major layout trick
    // C = alpha * (B * A) + beta * C  (Column Major view)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                dB, N, // Matrix B first
                dA, N, // Matrix A second
                &beta, 
                dC, N);

    // Record Start
    cudaEventRecord(start);

    // Actual Benchmark Run
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, dB, N, dA, N, &beta, dC, N);

    // Record Stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;

    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;
}