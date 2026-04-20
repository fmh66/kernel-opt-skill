#include <cuda_runtime.h>
#include <cfloat>

static __device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// v3: 3-pass (preserves L1 cache reuse) + float4 vectorized I/O + __expf fast math
//   Pass 1: float4 loads → row max
//   Pass 2: float4 loads → exp, float4 stores, accumulate sum
//   Pass 3: float4 loads from L1-cached output → multiply inv_sum, float4 stores
template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void softmax_vec4_fast(
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

    __shared__ float smem[NUM_WARPS];

    const int D4 = D >> 2;
    const float4* in4  = reinterpret_cast<const float4*>(input  + (long long)row * D);
    float4*       out4 = reinterpret_cast<float4*>      (output + (long long)row * D);

    // Pass 1: find row max via float4 loads
    float tmax = -FLT_MAX;
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 v = __ldg(in4 + i);
        tmax = fmaxf(tmax, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }

    tmax = warp_reduce_max(tmax);
    if (laneId == 0) smem[warpId] = tmax;
    __syncthreads();
    tmax = (tid < NUM_WARPS) ? smem[tid] : -FLT_MAX;
    if (warpId == 0) tmax = warp_reduce_max(tmax);
    if (tid == 0) smem[0] = tmax;
    __syncthreads();
    const float row_max = smem[0];

    // Pass 2: compute exp(x - max) with fast math, store float4, accumulate sum
    float tsum = 0.0f;
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 v = __ldg(in4 + i);
        float4 e;
        e.x = __expf(v.x - row_max);
        e.y = __expf(v.y - row_max);
        e.z = __expf(v.z - row_max);
        e.w = __expf(v.w - row_max);
        out4[i] = e;
        tsum += (e.x + e.y) + (e.z + e.w);
    }

    tsum = warp_reduce_sum(tsum);
    if (laneId == 0) smem[warpId] = tsum;
    __syncthreads();
    tsum = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
    if (warpId == 0) tsum = warp_reduce_sum(tsum);
    if (tid == 0) smem[0] = tsum;
    __syncthreads();
    const float inv_sum = __fdividef(1.0f, smem[0]);

    // Pass 3: normalize, reading back exp values from L1-cached output
    for (int i = tid; i < D4; i += BLOCK_SIZE) {
        float4 e = out4[i];
        e.x *= inv_sum;
        e.y *= inv_sum;
        e.z *= inv_sum;
        e.w *= inv_sum;
        out4[i] = e;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    constexpr int BLOCK_SIZE = 256;
    softmax_vec4_fast<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
