#include<iostream>
#include<cuda_runtime.h>

__global__ void naive_mm(float *a, float *b, float *c,
                         int m, int n, int k) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float p = 0.0f;
        for (int i = 0; i < k; i++) {
            p += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = p;
    }
}

int main() {

    // Square matrix for simplicity
    int N = 4096;
    int m = N, n = N, k = N;

    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float *A = (float*)malloc(sizeA);
    float *B = (float*)malloc(sizeB);
    float *C = (float*)malloc(sizeC);

    for (int i = 0; i < m * k; i++) A[i] = 1.0f;
    for (int i = 0; i < k * n; i++) B[i] = 1.0f;

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeA);
    cudaMalloc(&B_d, sizeB);
    cudaMalloc(&C_d, sizeC);

    cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    naive_mm<<<grid, block>>>(A_d, B_d, C_d, m, n, k); //warm-up

    cudaEventRecord(start);
    naive_mm<<<grid, block>>>(A_d, B_d, C_d, m, n, k);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost);

    // GFLOPS calculation
    double flops = 2.0 * N * N * N;   // 2*N^3
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    std::cout << "Time: " << ms << " ms\n";
    std::cout << "GFLOPS: " << gflops << std::endl;
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);

    return 0;
}

