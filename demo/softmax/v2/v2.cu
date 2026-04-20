#include <cuda_runtime.h>
#include <cfloat>

// v2: cache the full row in shared memory to reduce DRAM passes from 5 to 2
// (1 read of input + 1 write of output).
// float4 vectorized loads/stores for better bus utilization.
__global__ void softmax_v2(const float* __restrict__ input, float* __restrict__ output,
                           int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float srow[];   // D floats per block

    const float4* in4  = reinterpret_cast<const float4*>(input + (long)row * D);
    float4*       out4 = reinterpret_cast<float4*>(output + (long)row * D);
    float4*       sr4  = reinterpret_cast<float4*>(srow);

    int D4 = D >> 2;   // D / 4

    // ---- load row into shared memory (float4) ----
    for (int i = threadIdx.x; i < D4; i += blockDim.x)
        sr4[i] = in4[i];
    __syncthreads();

    // ---- pass 1: max from shared memory ----
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        max_val = fmaxf(max_val, srow[i]);

    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));

    __shared__ float smem[32];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int n_warps = blockDim.x >> 5;

    if (lid == 0) smem[wid] = max_val;
    __syncthreads();
    if (wid == 0) {
        max_val = (lid < n_warps) ? smem[lid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));
        if (lid == 0) smem[0] = max_val;
    }
    __syncthreads();
    max_val = smem[0];

    // ---- pass 2: exp in shared memory, accumulate sum ----
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float e = expf(srow[i] - max_val);
        srow[i] = e;
        sum_val += e;
    }

    // warp reduce sum
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

    // ---- pass 3: normalize shared memory & store to DRAM (float4) ----
    float inv_sum = __fdividef(1.0f, sum_val);
    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 v = sr4[i];
        v.x *= inv_sum; v.y *= inv_sum; v.z *= inv_sum; v.w *= inv_sum;
        out4[i] = v;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    int block     = 256;
    int smem_bytes = D * sizeof(float);
    softmax_v2<<<N, block, smem_bytes>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
