#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// v1: Parallelize dot-product & softmax across all d_k threads.
// Eliminates the if(t==0) serial bottleneck in v0.
// smem layout: [q_sm: d_k | scores: N | warp_buf: 32]
__global__ void mha_v1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N, int d_model, int h, int d_k)
{
    int head = blockIdx.x;
    int i    = blockIdx.y;
    int t    = threadIdx.x;

    int num_warps = (d_k + 31) >> 5;
    int warp_id   = t >> 5;
    int lane      = t & 31;

    extern __shared__ float smem[];
    float* q_sm    = smem;
    float* scores  = smem + d_k;
    float* warp_buf = smem + d_k + N;

    // Load Q[i, head, :] cooperatively
    q_sm[t] = Q[i * d_model + head * d_k + t];
    __syncthreads();

    // Each thread computes scores for its j positions (j = t, t+d_k, ...)
    float scale = rsqrtf((float)d_k);
    for (int j = t; j < N; j += d_k) {
        const float* kptr = K + j * d_model + head * d_k;
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++)
            dot = __fmaf_rn(q_sm[d], kptr[d], dot);
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Parallel max
    float my_max = -FLT_MAX;
    for (int j = t; j < N; j += d_k)
        my_max = fmaxf(my_max, scores[j]);
    my_max = warpReduceMax(my_max);
    if (lane == 0) warp_buf[warp_id] = my_max;
    __syncthreads();
    if (t == 0) {
        float gmax = warp_buf[0];
        for (int w = 1; w < num_warps; w++) gmax = fmaxf(gmax, warp_buf[w]);
        warp_buf[0] = gmax;
    }
    __syncthreads();
    float global_max = warp_buf[0];

    // Exp + parallel sum
    float my_sum = 0.0f;
    for (int j = t; j < N; j += d_k) {
        float e = expf(scores[j] - global_max);
        scores[j] = e;
        my_sum += e;
    }
    __syncthreads();

    my_sum = warpReduceSum(my_sum);
    if (lane == 0) warp_buf[warp_id] = my_sum;
    __syncthreads();
    if (t == 0) {
        float gs = warp_buf[0];
        for (int w = 1; w < num_warps; w++) gs += warp_buf[w];
        warp_buf[0] = gs;
    }
    __syncthreads();
    float inv_sum = __fdividef(1.0f, warp_buf[0]);

    // Normalize
    for (int j = t; j < N; j += d_k)
        scores[j] *= inv_sum;
    __syncthreads();

    // Output: thread t accumulates output[i, head, t]
    float val = 0.0f;
    for (int j = 0; j < N; j++)
        val = __fmaf_rn(scores[j], V[j * d_model + head * d_k + t], val);
    output[i * d_model + head * d_k + t] = val;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;
    dim3 grid(num_heads, N);
    dim3 block(d_k);
    size_t smem = (size_t)(d_k + N + 32) * sizeof(float);
    mha_v1_kernel<<<grid, block, smem>>>(Q, K, V, output, N, d_model, num_heads, d_k);
    cudaDeviceSynchronize();
}
