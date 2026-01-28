#include<iostream>
#include<cuda_runtime.h>
#include <torch/extension.h>

//input x -> (b, s, d) -> (b * s, d)
__global__ void layer_norm(float *x, float *out, float *weight, float *bias, int s, int d, float eps){

    int idx = threadIdx.x;
    int bs_offset = blockIdx.x;

    // int b_idx = bs_offset / s;
    // int s_idx = bs_offset % s; 

    // int offset = b_idx * (s * d) + s_idx * d;
    int offset = bs_offset * d;

    extern float __shared__ smem[];
    float *smem_sum = smem;
    float *smem_sqsum = smem_sum + blockDim.x;

    float sq_sum = 0.0f;
    float sum = 0.0f;
    for(int i = idx; i < d; i += blockDim.x){
      float curr_value = x[offset + i];
      sum += curr_value;
      sq_sum += (curr_value * curr_value);
    }
    smem_sum[idx] = sum;
    smem_sqsum[idx] = sq_sum;
    __syncthreads();

    for(int i = blockDim.x/2; i > 0 ; i /= 2){
      if(idx < i){
        smem_sum[idx] = smem_sum[idx] + smem_sum[idx + i];
        smem_sqsum[idx] = smem_sqsum[idx] + smem_sqsum[idx + i];
      }
      __syncthreads();
    }
    
    float global_sum = smem_sum[0];
    float global_sqsum = smem_sqsum[0];

    float mean = global_sum / d;
    float var = (global_sqsum / d) - (mean * mean);

    for(int i = idx; i < d; i += blockDim.x){
      float curr_value = x[offset + i];
      float out_value = (curr_value - mean)/(sqrtf(var + eps)); 
      out[offset + i] = out_value * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_forward(torch::Tensor x, float eps){

  int b = x.size(0);
  int s = x.size(1);
  int d = x.size(2);

  
  dim3 grid(b*s);
  dim3 blocks(256);

  int smem_size = 2 * blocks.x * sizeof(float);

  torch::Device device(torch::kCUDA);
  auto out = torch::empty({b, s, d}, device);
  auto weight = torch::ones({d}, device);
  auto bias = torch::zeros({d}, device);

  layer_norm<<<grid, blocks, smem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    weight.data_ptr<float>(),
    bias.data_ptr<float>(),
    s,
    d,
    eps
  );
  return out;
}