#include<iostream>
#include <cmath>
#include<cuda_runtime.h>

#define tile 16
#define coarse_factor 8

__global__ void shared_mm(float *a, float *b, float *c,
                         int m, int n, int k) {
    
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int row = blockIdx.y * blockDim.y + idy;
    int col = blockIdx.x * blockDim.x + idx;
    
    extern __shared__ float smem[];

    float *smemA = smem;
    float *smemB = smemA + tile*tile*coarse_factor; 
    
    float reg[coarse_factor][coarse_factor] = {0.0f};

    for(int i = 0; i < ((k + tile - 1)/tile); i ++){

      for(int l = 0; l < coarse_factor; l ++){

        int rowA = row + l*blockDim.y;
        int srowA = idy + l*blockDim.y;

        if(rowA< m && (i*tile+idx)<k){
          smemA[srowA * tile + idx] = a[rowA * k + i * tile + idx];
        }
        else{
          smemA[srowA * tile + idx] = 0.0f;
        }
      }

      for(int l = 0; l < coarse_factor; l++){

        int colB = col + l * blockDim.x;
        int scolB = idx + l * blockDim.x;

        if(colB < n && (i * tile + idy) < k){
          smemB[idy * (tile * coarse_factor) + scolB] = b[colB + (i * tile + idy) * n];
        }
        else{
          smemB[idy * (tile * coarse_factor) + scolB] = 0.0f;
        }
      }

      __syncthreads();

      #pragma unroll
      for(int p =0; p<coarse_factor; p ++){
        for(int q =0; q<coarse_factor; q ++ ){
          float local_sum = 0.0f;
          for(int j = 0; j < tile ; j++){
              local_sum += smemA[(idy + p*blockDim.y) * tile + j] * smemB[j * (tile * coarse_factor) + (idx+q*blockDim.x)];
          }
          reg[p][q] += local_sum;
        }
      }
      __syncthreads();
    }

    for(int p =0; p<coarse_factor; p ++){
      for(int q =0; q<coarse_factor; q ++){

        int rowC = row + p * blockDim.y;
        int colC = col + q * blockDim.x;

        if(rowC < m && colC < n){
          c[rowC * n + colC] = reg[p][q];
        }
      }
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
    dim3 grid((N + tile*coarse_factor - 1) / (tile*coarse_factor),
              (N + tile*coarse_factor - 1) / (tile*coarse_factor));

    size_t shared_bytes = 2 * tile * (tile * coarse_factor) * sizeof(float);

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