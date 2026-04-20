#include <cuda_runtime.h>
#include <float.h>

// One block per row: coalesced access + warp shuffle reduction.
// v0 assigned one thread per row with stride-D access → 12.5% efficiency.
// Here each block cooperates on a row: threads access consecutive elements,
// warps reduce with __shfl_xor_sync, final reduce via shared memory.
__global__ void softmax_v1(float* __restrict__ input,
                            float* __restrict__ output,
                            int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* __restrict__ in_row  = input  + (long long)row * D;
    float*       __restrict__ out_row = output + (long long)row * D;

    extern __shared__ float smem[];  // [num_warps] floats

    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // --- pass 1: find row max ---
    float max_val = -FLT_MAX;
    for (int i = tid; i < D; i += blockDim.x)
        max_val = fmaxf(max_val, in_row[i]);

    // warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));

    if (lane_id == 0) smem[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
        if (lane_id == 0) smem[0] = max_val;
    }
    __syncthreads();
    max_val = smem[0];

    // --- pass 2: compute exp and partial sum ---
    float sum_val = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = expf(in_row[i] - max_val);
        out_row[i] = val;
        sum_val += val;
    }

    // warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, offset);

    if (lane_id == 0) smem[warp_id] = sum_val;
    __syncthreads();

    if (warp_id == 0) {
        sum_val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_val += __shfl_xor_sync(0xffffffff, sum_val, offset);
        if (lane_id == 0) smem[0] = sum_val;
    }
    __syncthreads();
    sum_val = smem[0];

    // --- pass 3: normalize ---
    float inv_sum = 1.0f / sum_val;
    for (int i = tid; i < D; i += blockDim.x)
        out_row[i] *= inv_sum;
}

extern "C" void solve(float* input, float* output, int N, int D) {
    int threads   = 128;
    int num_warps = threads / 32;
    int shmem     = num_warps * sizeof(float);
    softmax_v1<<<N, threads, shmem>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
