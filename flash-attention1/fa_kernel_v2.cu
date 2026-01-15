#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>


__global__ void flash_attention (float *q, float *k, float *v, float *o, float *m, float *l, 
  int Tr, int Tc, int Bc, int Br, int Nh, int s, int d, float scale){
  
  extern __shared__ float smem[];
  
  float *smem_k = smem;
  float *smem_v = smem_k + Bc * d;
  float *smem_q = smem_v + Bc * d;
  float *smem_sij = smem_q + Br * d;
  float *smem_o = smem_sij + Br * Bc;
  float *smem_l = smem_o + Br * d;
  float *smem_m = smem_l + Br; 
  float *smem_lnew = smem_m + Br; 
  float *smem_mnew = smem_lnew + Br;
  float *smem_mlocal = smem_mnew + Br; 

  int b_idx = blockIdx.x;
  int Nh_idx = blockIdx.y;

  int qkv_off = b_idx * (Nh * s * d) + Nh_idx * (s * d);
  int lm_off = b_idx * (Nh * s) + Nh_idx * (s);

  for(int j = 0; j < Tc; j++){

    //loading Bc * d tile for K and V
    int y = threadIdx.x / Bc;
    int x = threadIdx.x % Bc;

    int row = j * Bc + y;
    for(int c = 0; c < (d + Br - 1) / Br ; c++){
      if(row < s && (x + Br * c) < d){
        smem_k[y * d + (x + Br * c)] = k[qkv_off + row * d + (x + Br * c)];
        smem_v[y * d + (x + Br * c)] = v[qkv_off + row * d + (x + Br * c)];
      }
    }

    __syncthreads(); 

    for(int i = 0; i < Tr; i++){
      //loading the Q tile -> (Br * d)
      int y = threadIdx.x / Br;
      int x = threadIdx.x % Br;

      int row = i * Br + y;
      for(int c = 0; c < (d + Bc - 1) / Bc ; c++){
        if(row < s && (x + Bc * c) < d){
          smem_q[y * d + (x + Bc * c)] = q[qkv_off + row * d + (x + Bc * c)];
          smem_o[y * d + (x + Bc * c)] = o[qkv_off + row * d + (x + Bc * c)];
        }
      }

      //loading the l and m
      if(x==0){
        if(row < s){
          smem_l[y] = l[lm_off + row];
          smem_m[y] = m[lm_off + row];
        }
      }

      __syncthreads();

      //computing the dot product
      float sij = 0.0f;
      for(int k = 0; k < d; k++){
        sij += smem_q[y * d + k] * smem_k[x * d + k];
      }
      sij *= scale;
      smem_sij[y * Bc + x] = sij;
      
      __syncthreads();

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

        for (int c = 0; c < Bc; c++) {
          float exp_val = expf(smem_sij[y * Bc + c] - local_max);
          smem_sij[y * Bc + c] = exp_val;
        }

        smem_mlocal[y] = local_max; 
        smem_mnew[y] = fmaxf(smem_m[y], local_max);
        smem_lnew[y] = expf(smem_m[y] - smem_mnew[y]) * smem_l[y] + local_sum * expf(local_max - smem_mnew[y]);

      }
      __syncthreads();

        for(int c = 0; c < (d + Bc - 1) / Bc; c++){

        float p = 0.0f;
        for(int k = 0; k < Bc; k++){
          p += smem_sij[y * Bc + k] * smem_v[k * d + (x + c * Bc)];
        }

        smem_o[y * d + (x + Bc * c)] = (smem_l[y] * smem_o[y * d + (x + Bc * c)] * expf(smem_m[y] - smem_mnew[y])
                                  + expf(smem_mlocal[y] - smem_mnew[y]) * p)/smem_lnew[y];
        o[qkv_off + row * d + (x + Bc * c)] = smem_o[y * d + (x + Bc * c)];

      }

    l[lm_off + row] = smem_lnew[y];
    m[lm_off + row] = smem_mnew[y];

    
    }
    __syncthreads();
  }
}

torch:: Tensor fa_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    constexpr int Br = 16;
    constexpr int Bc = 16;

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
    auto m = torch::full({B, Nh, T}, -FLT_MAX, device);

    dim3 block(Br * Bc);
    dim3 grid(B, Nh);
    
    size_t smem_bytes =
        (Bc * d +   // K
         Bc * d +   // V
         Br * d +   // Q
         Br * d  +  // O
         Br * Bc +    // Sij
         5 * (Br)
        ) * sizeof(float);

    flash_attention<<<grid, block, smem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        o.data_ptr<float>(),
        m.data_ptr<float>(),
        l.data_ptr<float>(),
        Tr,
        Tc,
        Bc,
        Br,
        Nh,
        T,
        d,
        softmax_scale
    );
    return o;
}