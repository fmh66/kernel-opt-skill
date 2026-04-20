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

// One block per row; coalesced loads; warp-shuffle + shared-mem reduction
template <int BLOCK_SIZE>
__global__ void softmax_block_per_row(
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

    const float* in_row  = input  + (long long)row * D;
    float*       out_row = output + (long long)row * D;

    // Pass 1: thread-local max
    float tmax = -FLT_MAX;
    for (int i = tid; i < D; i += BLOCK_SIZE)
        tmax = fmaxf(tmax, __ldg(in_row + i));

    // Warp reduce → block reduce
    tmax = warp_reduce_max(tmax);
    if (laneId == 0) smem[warpId] = tmax;
    __syncthreads();
    tmax = (tid < NUM_WARPS) ? smem[tid] : -FLT_MAX;
    if (warpId == 0) tmax = warp_reduce_max(tmax);
    if (tid == 0) smem[0] = tmax;
    __syncthreads();
    const float row_max = smem[0];

    // Pass 2: exp(x - max), write to output, accumulate sum
    float tsum = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float val = expf(__ldg(in_row + i) - row_max);
        out_row[i] = val;
        tsum += val;
    }

    // Warp reduce → block reduce
    tsum = warp_reduce_sum(tsum);
    if (laneId == 0) smem[warpId] = tsum;
    __syncthreads();
    tsum = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
    if (warpId == 0) tsum = warp_reduce_sum(tsum);
    if (tid == 0) smem[0] = tsum;
    __syncthreads();
    const float inv_sum = __fdividef(1.0f, smem[0]);

    // Pass 3: normalize
    for (int i = tid; i < D; i += BLOCK_SIZE)
        out_row[i] *= inv_sum;
}

extern "C" void solve(float* input, float* output, int N, int D) {
    constexpr int BLOCK_SIZE = 256;
    softmax_block_per_row<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
