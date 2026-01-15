#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

//approach
//block x -> B * Nh, block y -> S/Br  
// threads in block -> Br * Bc

__global__ void flash_attention(float *q, float *k, float *v, float *o, float *l, int Nh, int s, int d,
   int Tr, int Tc, int Br, int Bc, float scale){
    
    extern float __shared__ smem[];

    int pad = 8;
    int d_padded = d + pad;

    //smem loading
    float *smem_q = smem;
    float *smem_o = smem_q + Br * d_padded;
    float *smem_k = smem_o + Br * d_padded;
    float *smem_v = smem_k + Bc * d_padded;
    float *smem_sij = smem_v + Bc * d_padded;
    float *smem_mj = smem_sij + Br * Bc;
    float *smem_lj = smem_mj + Br;
    // float *smem_locm = smem_lj + Br;

    int B_Nh = blockIdx.x;
    int S_idx = blockIdx.y;
    int idx = threadIdx.x;

    int B_idx = B_Nh / Nh;
    int Nh_idx = B_Nh % Nh;
    int qkv_offset = B_idx * (Nh * s * d) + Nh_idx * (s * d);
    int lm_offset = B_idx * (Nh * s) + Nh_idx * (s);

    //loading the q, l and m and o
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    for (int i = idx; i < Br * d; i += blockDim.x) {
      smem_o[i] = 0.0f;
    }
    __syncthreads();

    int x = idx % Bc;
    int y = idx / Bc;
    
    int row_q = S_idx * Br + y;
    for(int c = 0; c < (Bc + d -1)/ Bc; c++){
      if(row_q < s && (x + Bc * c) < d){
        smem_q[y * d_padded + (x + Bc * c)] = q[qkv_offset + row_q * d + (x + Bc * c)];
      }
    }

    for(int j = 0; j < Tc; j++){

      //loading the k and v
      int xj = idx % Br;
      int yj = idx / Br;

      int row_kv = j * Bc + yj;
      for(int c = 0; c < (Br + d -1)/ Br; c++){
      if(row_kv < s && (xj + Br * c) < d){
          smem_k[yj * d_padded + (xj + Br * c)] = k[qkv_offset + row_kv * d + (xj + Br * c)];
          smem_v[yj * d_padded + (xj + Br * c)] = v[qkv_offset + row_kv * d + (xj + Br * c)];
        }
      }
      __syncthreads();

      float sij = 0.0f;
      for(int k = 0; k < d; k++){
        sij += smem_q[y * d_padded + k] * smem_k[x * d_padded + k];
      }
      sij *= scale;
      smem_sij[y * Bc + x] = sij;

      __syncthreads();

      //online softmax
      if(x==0){
        float local_max = -FLT_MAX;
        float local_sum = 0.0f; 
        for(int k = 0; k < Bc; k++){
          float curr_value = smem_sij[y * Bc + k];
          if(curr_value > local_max){
            local_sum *= expf(local_max - curr_value);
            local_max = curr_value;
          }
          local_sum += expf(curr_value - local_max);
        }

        smem_mj[y] = fmaxf(row_max, local_max);
        smem_lj[y] = row_sum * expf(row_max - smem_mj[y]) + local_sum * expf(local_max - smem_mj[y]);
      }
      
      __syncthreads();
      
      //calcualting the P
      for(int c = 0; c < (Bc + d -1)/ Bc; c++){

        float p = 0.0f;
        for(int k = 0; k < Bc; k++){
          p += expf(smem_sij[y * Bc + k] - smem_mj[y]) * smem_v[k * d_padded + (x + Bc * c)];
        }
        smem_o[y * d_padded + (x + Bc * c)] = expf(row_max - smem_mj[y]) * smem_o[y * d_padded + (x + Bc * c)] + p;
      }

      row_max = smem_mj[y];
      row_sum = smem_lj[y];
      
      __syncthreads();
    }

    //writing back o and l
    for(int c = 0; c < (Bc + d -1)/ Bc; c++){
      if(row_q < s && (x + Bc * c) < d){
        o[qkv_offset + row_q * d + (x + Bc * c)] = smem_o[y * d_padded + (x + Bc * c)]/row_sum;
      }
    }

    l[lm_offset + row_q] = row_max + logf(row_sum);

}

torch:: Tensor fa_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    constexpr int Br = 16;
    constexpr int Bc = 8;

    const int B = q.size(0);
    const int Nh = q.size(1);
    const int T = q.size(2);
    const int d = q.size(3);

    const int Tr = (T + Br - 1) / Br;
    const int Tc = (T + Bc - 1) / Bc;

    float softmax_scale = 1.0 / sqrt(d);
    
    torch::Device device(torch::kCUDA);
    auto o = torch::zeros_like(q);
    auto l = torch::zeros({B, Nh, T}, device);

    
    size_t smem_bytes =
      (Bc * (d + 8) +   // K
      Bc * (d + 8) +   // V
      Br * (d + 8) +   // Q
      Br * (d + 8)  +  // O
      Br * Bc +    // Sij
      2 * (Br)
      ) * sizeof(float);
    
    // int max_sram_size;
    // cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_bytes);

    dim3 block(Br * Bc);
    dim3 grid(B * Nh, Tr);

    flash_attention<<<grid, block, smem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        o.data_ptr<float>(),
        l.data_ptr<float>(),
        Nh,
        T,
        d,
        Tr,
        Tc,
        Br,
        Bc,
        softmax_scale
    );
    return o;
}
