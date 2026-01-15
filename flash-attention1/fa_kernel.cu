#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>


__global__ void flash_attention (float *q, float *k, float *v, float *o, float *m, float *l, int Tr, int Tc, int Bc, int Br, int d){

    int idx = threadIdx.x; // represent Br
    int idy = threadIdx.y; // represent Bc

    
    int col = blockIdx.x * blockDim.x + idx; 

    extern __shared__ float smem[];

    float *smem_k = smem;
    float *smem_v = smem_k + Bc * d;
    float *smem_q = smem_v + Bc * d;
    float *smem_sij = smem_q + Br * d;
    float *smem_o = smem_sij + Br * Bc;
    float *smem_int = smem_o + Br * d;

    float l_i = 0.0f;
    float m_i = -INFINITY;
    for(int k=0; k< Br * d; ++k) smem_o[k] = 0.0f;

    for(int c = 0; c < (d + Bc - 1) / Bc; c++){
        smem_q[idx * d + (idy + Bc * c)] = q[col * d + (idy + Bc * c)];
    } //(????? check this later it is uncoalesced)


    for(int j = 0; j < Tc; j++){

      int row = j * blockDim.y + idy;
      for(int c = 0; c < (d + Br - 1) / Br ; c++){
        smem_k[idy * d + (idx + Br * c)] = k[row * d + (idx + Br * c)];
      }

      for(int c = 0; c < (d + Br - 1) / Br; c++){
        smem_v[idy * d + (idx + Br * c)] = v[row * d + (idx + Br * c)];
      }

      __syncthreads(); 

      //Computing the Sij
      float sij = 0.0f;
      for(int k = 0; k < d; k++){
        sij += smem_q[idx * d + k] * smem_k[idy * d + k];
      }

      smem_sij[idx * Bc + idy] = sij; //Br*Bc

      __syncthreads();

      float local_max = -INFINITY;
      float local_sum = 0.0f;
      for(int k = 0; k < Bc; k++){
        float curr_value = smem_sij[idx * Bc + k];
        if(curr_value > local_max){
          local_sum *= expf(local_max - curr_value);
          local_max = curr_value;
        }
        local_sum += expf(curr_value - local_max);
      }

      float max_new = fmaxf(m_i, local_max);
      float sum_new = expf(m_i - max_new) * l_i + local_sum * expf(local_max - max_new);

      for(int c = 0; c < (d + Bc - 1) / Bc; c++){
        float p = 0.0f;
        for(int k = 0; k < Bc; k++){
            p += expf(smem_sij[idx * Bc + k] - local_max) * smem_v[k * d + (idy + c * Bc)]; //Bc * d
        }
        smem_int[idx * d + (idy + c * Bc)] = p; //Br*d
      }

      for(int c = 0; c < (d + Bc - 1) / Bc; c++){
        smem_o[idx * d + (idy + Bc * c)] = (l_i * smem_o[idx * d + (idy + Bc * c)] * expf(m_i - max_new)
                                        + expf(local_max - max_new) * smem_int[idx * d + (idy + Bc * c)])/sum_new;
      }

      m_i = max_new;
      l_i = sum_new;

    }

    for(int k=0; k<d; k++){
      o[col * d + k] = smem_o[idx * d + k];
    }

    m[col] = m_i;
    l[col] = l_i;

}


torch:: Tensor fa_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    constexpr int Br = 32;
    constexpr int Bc = 32;

    const int T = q.size(0);
    const int d = q.size(1);

    const int Tr = ceil(T/Br);
    const int Tc = ceil(T/Bc);

    float softmax_scale = 1.0 / sqrt(d);

    auto o = torch::zeros_like(q);
    auto l = torch::zeros({T});
    auto m = torch::full({T}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);


    dim3 block(Br, Bc);
    dim3 grid(
        (T + Br - 1) / Br,
        (T + Bc - 1) / Bc
    );
    
    size_t smem_bytes =
        (Bc * d +   // K
         Bc * d +   // V
         Br * d +   // Q
         Br * Bc +  // Sij
         Br * d  +  // intermediate O
         Br * d     // intermediate output
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
        d
    );
    return o;
}