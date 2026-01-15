#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>

//batch, heads, sequence, head_dim/2
// grid will be launched like this -> (batch * heads, (sequence * head_dim/2)//(idx * idy))
// block will be (idx * idy)

__global__ void rope_kernel(float *x, float *rope_freq, int Nh, int s, int d){

    int half_d = d / 2;
    int B_Nh = blockIdx.x;

    int B_idx = B_Nh / Nh;
    int Nh_idx = B_Nh % Nh;

    int t_threads = blockDim.x;
    int idx = threadIdx.x;

    //for the sequence and half_d
    int global_idx = blockIdx.y * t_threads + idx;

    int S_idx = global_idx / half_d;
    int d_idx = global_idx % half_d;

    int b_Nh_offset = B_idx * (Nh * s * d) + Nh_idx * (s * d);

    if (S_idx >= s || d_idx >= half_d) {
        return;
    }

    float value_1 = x[b_Nh_offset + S_idx * d + d_idx];
    float value_2 = x[b_Nh_offset + S_idx * d + (d_idx + half_d)];

    //loading the rope_freq
    float m_theta = rope_freq[S_idx * half_d + d_idx];

    //x_i' = x_i * cos(m_Theta_i) - x_{i + D/2} * sin(m_Theta_i)
    float n_value1 = value_1 * cosf(m_theta) - value_2 * sinf(m_theta);

    //x'_{i + D/2} = x_i * sin(m_Theta_i) + x_{i + D/2} * cos(m_Theta_i)
    float n_value2 = value_1 * sinf(m_theta) + value_2 * cosf(m_theta);

    x[b_Nh_offset + S_idx * d + d_idx] = n_value1;
    x[b_Nh_offset + S_idx * d + (d_idx + half_d)] = n_value2;

}

void rope_cuda_forward(torch::Tensor x, torch::Tensor freqs) {

    int B = x.size(0);
    int Nh = x.size(1);
    int S = x.size(2);
    int D = x.size(3);
    
    int half_d = D / 2;
    int total_elements = S * half_d;
    int t_threads = 32 * 32; //blockdim.x

    int grid_x = B * Nh;
    int grid_y = (total_elements +  t_threads - 1) / t_threads;

    dim3 blocks(grid_x, grid_y);
    dim3 threads(32 * 32);

    rope_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        freqs.data_ptr<float>(), 
        Nh, S, D
    );
}
