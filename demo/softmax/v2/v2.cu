#include <cuda_runtime.h>
#include <cfloat>

static __device__ __forceinline__ void warp_reduce_online(float& m, float& s) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, offset);
        float os = __shfl_down_sync(0xffffffff, s, offset);
        float new_m = fmaxf(m, om);
        s = s * expf(m - new_m) + os * expf(om - new_m);
        m = new_m;
    }
}

// Online softmax (Milakov & Gimelshein): 2 global passes instead of 3
//   Pass 1: scan input once, accumulate running (max, sum_exp) — no output writes
//   Pass 2: scan input once, write exp(x-max)/sum to output
// Also uses float4 vectorized loads for better L1/L2 throughput.
template <int BLOCK_SIZE>
__global__ void softmax_online_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int D)
{
    const int row = blockIdx.x;
    if (row >= N) return;

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int tid    = threadIdx.x;
    const int warpId = tid >> 5;
    const int laneId = tid & 31;

    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_s[NUM_WARPS];

    const int D4 = D >> 2;  // D must be divisible by 4
    const float4* in4  = reinterpret_cast<const float4*>(input  + (long long)row * D);
    float4*       out4 = reinterpret_cast<float4*>      (output + (long long)row * D);

    // Pass 1: accumulate online (max, sum_exp) in one scan of input
    float tmax = -FLT_MAX, tsum = 0.0f;
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 v = __ldg(in4 + i);
        #pragma unroll
        for (float x : {v.x, v.y, v.z, v.w}) {
            float new_m = fmaxf(tmax, x);
            tsum = tsum * expf(tmax - new_m) + expf(x - new_m);
            tmax = new_m;
        }
    }

    // Warp-level reduce
    warp_reduce_online(tmax, tsum);
    if (laneId == 0) { smem_m[warpId] = tmax; smem_s[warpId] = tsum; }
    __syncthreads();

    // Block-level reduce (warp 0 only)
    if (warpId == 0) {
        tmax = (tid < NUM_WARPS) ? smem_m[tid] : -FLT_MAX;
        tsum = (tid < NUM_WARPS) ? smem_s[tid] : 0.0f;
        warp_reduce_online(tmax, tsum);
        if (tid == 0) { smem_m[0] = tmax; smem_s[0] = tsum; }
    }
    __syncthreads();

    const float row_max = smem_m[0];
    const float inv_sum = __fdividef(1.0f, smem_s[0]);

    // Pass 2: write exp(x - max) * inv_sum using float4
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 v = __ldg(in4 + i);
        float4 o;
        o.x = expf(v.x - row_max) * inv_sum;
        o.y = expf(v.y - row_max) * inv_sum;
        o.z = expf(v.z - row_max) * inv_sum;
        o.w = expf(v.w - row_max) * inv_sum;
        out4[i] = o;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    constexpr int BLOCK_SIZE = 256;
    softmax_online_vec4<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
