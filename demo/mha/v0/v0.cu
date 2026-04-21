#include <cuda_runtime.h>
#include <math.h>

__global__ void multi_head_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int N, int d_model, int h, int d_k)
{
    int head = blockIdx.x;
    int i    = blockIdx.y;
    int t    = threadIdx.x;

    if (t >= d_k) return;

    extern __shared__ float scores[];

    if (t == 0) {
        for (int j = 0; j < N; j++) {
            const float* q_ptr = Q + i * d_model + head * d_k;
            const float* k_ptr = K + j * d_model + head * d_k;
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += q_ptr[d] * k_ptr[d];
            }
            scores[j] = dot / sqrtf((float)d_k);
        }

        float max_val = scores[0];
        for (int j = 1; j < N; j++) {
            max_val = fmaxf(max_val, scores[j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - max_val);
            sum += scores[j];
        }

        for (int j = 0; j < N; j++) {
            scores[j] /= sum;
        }
    }

    __syncthreads();

    float val = 0.0f;
    for (int j = 0; j < N; j++) {
        val += scores[j] * V[j * d_model + head * d_k + t];
    }

    output[i * d_model + head * d_k + t] = val;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;

    dim3 grid(num_heads, N);
    dim3 block(d_k);
    size_t shared_mem = N * sizeof(float);

    multi_head_attention_kernel<<<grid, block, shared_mem>>>(
        Q, K, V, output, N, d_model, num_heads, d_k);

    cudaDeviceSynchronize();
}