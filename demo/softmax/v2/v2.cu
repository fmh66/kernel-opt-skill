#include <cuda_runtime.h>
#include <float.h>

// v2: Online softmax (fused max+sum in one DRAM pass) + float4 vectorized loads.
// v1 made 3 DRAM passes: (1) read for max, (2) read+exp+sum, (3) read+normalize.
// v2 fuses passes 1+2 into one online pass, leaving only 2 DRAM reads + 1 write.
// Online merge rule for partial (m1,s1) and (m2,s2):
//   m = max(m1,m2),  s = s1*exp(m1-m) + s2*exp(m2-m)
__global__ void softmax_v2(float* __restrict__ input,
                            float* __restrict__ output,
                            int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float4* __restrict__ in4 = (const float4*)(input  + (long long)row * D);
    float4*       __restrict__ ot4 = (float4*)(output + (long long)row * D);

    extern __shared__ float smem[];  // [num_warps] max | [num_warps] sum
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int num_warps = blockDim.x >> 5;
    const int D4        = D >> 2;

    float* smax = smem;
    float* ssum = smem + num_warps;

    // --- Pass 1: Online (max, sum) over float4 chunks ---
    float row_max = -FLT_MAX, row_sum = 0.0f;

    for (int i = tid; i < D4; i += blockDim.x) {
        float4 v      = __ldg(&in4[i]);
        float lm      = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        float new_max = fmaxf(row_max, lm);
        row_sum = row_sum * expf(row_max - new_max)
                + expf(v.x - new_max) + expf(v.y - new_max)
                + expf(v.z - new_max) + expf(v.w - new_max);
        row_max = new_max;
    }

    // Warp reduce: online merge
    for (int off = 16; off > 0; off >>= 1) {
        float om = __shfl_xor_sync(0xffffffff, row_max, off);
        float os = __shfl_xor_sync(0xffffffff, row_sum, off);
        float nm = fmaxf(row_max, om);
        row_sum  = row_sum * expf(row_max - nm) + os * expf(om - nm);
        row_max  = nm;
    }
    if (lane_id == 0) { smax[warp_id] = row_max; ssum[warp_id] = row_sum; }
    __syncthreads();

    if (warp_id == 0) {
        row_max = (lane_id < num_warps) ? smax[lane_id] : -FLT_MAX;
        row_sum = (lane_id < num_warps) ? ssum[lane_id] :  0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float om = __shfl_xor_sync(0xffffffff, row_max, off);
            float os = __shfl_xor_sync(0xffffffff, row_sum, off);
            float nm = fmaxf(row_max, om);
            row_sum  = row_sum * expf(row_max - nm) + os * expf(om - nm);
            row_max  = nm;
        }
        if (lane_id == 0) { smax[0] = row_max; ssum[0] = row_sum; }
    }
    __syncthreads();

    row_max = smax[0];
    float inv_sum = 1.0f / ssum[0];

    // --- Pass 2: Normalize with float4 ---
    for (int i = tid; i < D4; i += blockDim.x) {
        float4 v = __ldg(&in4[i]);
        float4 o;
        o.x = expf(v.x - row_max) * inv_sum;
        o.y = expf(v.y - row_max) * inv_sum;
        o.z = expf(v.z - row_max) * inv_sum;
        o.w = expf(v.w - row_max) * inv_sum;
        ot4[i] = o;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    int threads   = 256;
    int num_warps = threads / 32;
    int shmem     = 2 * num_warps * sizeof(float);
    softmax_v2<<<N, threads, shmem>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
