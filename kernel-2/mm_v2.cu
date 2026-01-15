#include<iostream>
#include <cmath>
#include<cuda_runtime.h>

#define tile 16

__global__ void shared_mm(float *a, float *b, float *c,
                         int m, int n, int k) {
    
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int row = blockIdx.y * blockDim.y + idy;
    int col = blockIdx.x * blockDim.x + idx;
    
    extern __shared__ float smem[];

    float *smemA = smem;
    float *smemB = smemA + tile*tile; 
    
    float p = 0.0f;
    for(int i = 0; i < ((k + tile - 1)/tile); i ++){

      if(row < m && (i*tile+idx)<k){
        smemA[idy * tile + idx] = a[row * k + i * tile + idx];
      }
      else{
        smemA[idy * tile + idx] = 0.0f;
      }

      if(col < n && (i * tile + idy) < k){
        smemB[idy * tile + idx] = b[col + (i * tile + idy) * n];
      }
      else{
        smemB[idy * tile + idx] = 0.0f;
      }

      __syncthreads();
      
      #pragma unroll
      for(int j = 0; j < tile ; j++){
          p += smemA[idy * tile + j] * smemB[j * tile + idx];
      }
      __syncthreads();
    }

    if(row < m && col < n){
      c[row * n + col] = p;
    }
  }


int main() {

    int s = 4096;
    int M = s, N = s, K = s;

    size_t bytes = N * N * sizeof(float);

    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    float* hC_ref = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
        hC_ref[i] = 0.0f;
    }

    //cpu mm
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         float sum = 0.0f;
    //         for (int k = 0; k < N; k++) {
    //             sum += hA[i * N + k] * hB[k * N + j];
    //         }
    //         hC_ref[i * N + j] = sum;
    //     }
    // }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile,
              (N + tile - 1) / tile);

    size_t shared_bytes = 2 * tile * tile * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //warp up
    shared_mm<<<grid, block, shared_bytes>>>(dA, dB, dC, M, N, K);

    cudaEventRecord(start);
    shared_mm<<<grid, block, shared_bytes>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);


    // double max_error = 0.0;
    // for (int i = 0; i < N * N; i++) {
    //     double diff = std::abs(hC[i] - hC_ref[i]);
    //     max_error = std::max(max_error, diff);
    // }

    // if (max_error < 1e-3) {
    //     std::cout << " Result correct\n";
    // } else {
    //     std::cout << " Result incorrect, max error = " << max_error << "\n";
    // }

    double flops = 2.0 * N * N * N;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    std::cout << "Time: " << ms << " ms\n";
    std::cout << "GFLOPS: " << gflops << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    free(hC_ref);

    return 0;
}

