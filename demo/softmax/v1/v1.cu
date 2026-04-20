#include <cuda_runtime.h>
#include <cfloat>

// One block per row: improves occupancy (N blocks vs N/256 blocks)
// and enables coalesced memory access (consecutive threads -> consecutive elements in same row).
// Warp shuffle reduces max/sum without shared memory bank conflicts.
__global__ void softmax_v1(const float* __restrict__ input, float* __restrict__ output,
                           int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* in_row  = input  + (long)row * D;
    float*       out_row = output + (long)row * D;

    // ---- pass 1: find max (coalesced strided load) ----
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        max_val = fmaxf(max_val, in_row[i]);

    // warp-level reduce max
    for (int off = 16; off > 0; off >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));

    __shared__ float smem[32];
    int wid = threadIdx.x >> 5;   // threadIdx.x / 32
    int lid = threadIdx.x & 31;   // threadIdx.x % 32

    if (lid == 0) smem[wid] = max_val;
    __syncthreads();

    // block-level reduce max (first warp)
    int n_warps = blockDim.x >> 5;
    if (wid == 0) {
        max_val = (lid < n_warps) ? smem[lid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));
        if (lid == 0) smem[0] = max_val;
    }
    __syncthreads();
    max_val = smem[0];

    // ---- pass 2: compute exp, accumulate sum (write to output) ----
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum_val += e;
    }

    // warp-level reduce sum
    for (int off = 16; off > 0; off >>= 1)
        sum_val += __shfl_down_sync(0xffffffff, sum_val, off);

    if (lid == 0) smem[wid] = sum_val;
    __syncthreads();

    if (wid == 0) {
        sum_val = (lid < n_warps) ? smem[lid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum_val += __shfl_down_sync(0xffffffff, sum_val, off);
        if (lid == 0) smem[0] = sum_val;
    }
    __syncthreads();
    sum_val = smem[0];

    // ---- pass 3: normalize ----
    float inv_sum = __fdividef(1.0f, sum_val);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out_row[i] *= inv_sum;
}

extern "C" void solve(float* input, float* output, int N, int D) {
    softmax_v1<<<N, 256>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
