#include<iostream>
#include<cuda_runtime.h>
#include<torch/extension.h>

//(b, s, 2d) -> grid(b, ceil(s/32) * ceil(d / 32))
__device__ __forceinline__ float gelu(float x){
    float in = x / sqrtf(2);
    return 0.5f * x * (1.0f + erff(in));
}
__global__ void gated_gelu(float *x, float *out, int s, int D){

  int b_idx = blockIdx.x;
  int batch_offset = b_idx * (s * D);

  int d = D/2; //considering the D is divisible by 2

  int idx = threadIdx.x;
  int global_idx = idx + blockIdx.y * blockDim.x;
  int s_idx = global_idx / d;
  int d_idx = global_idx % d;

  if(s_idx >= s || d_idx >= d) return;
  //loading the elements
  float value_1 = x[batch_offset + s_idx * D + d_idx];
  float value_2 = x[batch_offset + s_idx * D + (d_idx + d)];

  float out_value = gelu(value_1) * (value_2 + 1.0f);
  out[b_idx * (s * d) + s_idx * d + d_idx] = out_value;

}

torch::Tensor gated_gelu_forward(torch::Tensor x) {

    int B = x.size(0);
    int S = x.size(1);
    int D = x.size(2);
    int half_d = D / 2;

    torch::Device device(torch::kCUDA);
    auto out = torch::empty({B,S,half_d}, device);
    
    int total_elements = S * half_d;
    int t_threads = 32 * 32; //blockdim.x

    int grid_x = B;
    int grid_y = (total_elements +  t_threads - 1) / t_threads;

    dim3 blocks(grid_x, grid_y);
    dim3 threads(32 * 32);

    gated_gelu<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        S, D
    );
    return out;
}
