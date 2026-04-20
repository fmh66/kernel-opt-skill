#include <cuda_runtime.h>
#include <float.h>

// v3: Shared-memory row cache + online softmax + float4.
// v2 still reads DRAM twice (online pass + normalize pass).
// v3 loads the row into shared memory once, then all compute uses shmem
// (latency ~20 cycles vs DRAM ~400 cycles), yielding minimum DRAM traffic:
//   1 x float4 DRAM read  (shmem fill)
//   1 x float4 DRAM write (normalized output)
// Online softmax merge rule: (m,s)+(m',s') -> (max(m,m'), s*exp(m-M)+s'*exp(m'-M))
__global__ __launch_bounds__(256, 4)
void softmax_v3(float* __restrict__ input,
                float* __restrict__ output,
                int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float4* __restrict__ in4 = (const float4*)(input  + (long long)row * D);
    float4*       __restrict__ ot4 = (float4*)(output + (long long)row * D);

    // smem layout: [D floats = row cache] [num_warps floats = wmax] [num_warps floats = wsum]
    extern __shared__ float smem[];
    const int num_warps = blockDim.x >> 5;
    float* row_cache = smem;
    float* wmax_buf  = smem + D;
    float* wsum_buf  = smem + D + num_warps;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int D4      = D >> 2;

    // --- Phase 0: Load row into shared memory via float4 ---
    float4* scache4 = (float4*)row_cache;
    for (int i = tid; i < D4; i += blockDim.x)
        scache4[i] = __ldg(&in4[i]);
    __syncthreads();

    // --- Phase 1: Online (max, sum) from shared memory ---
    float row_max = -FLT_MAX, row_sum = 0.0f;

    for (int i = tid; i < D4; i += blockDim.x) {
        float4 v      = scache4[i];
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
    if (lane_id == 0) { wmax_buf[warp_id] = row_max; wsum_buf[warp_id] = row_sum; }
    __syncthreads();

    if (warp_id == 0) {
        row_max = (lane_id < num_warps) ? wmax_buf[lane_id] : -FLT_MAX;
        row_sum = (lane_id < num_warps) ? wsum_buf[lane_id] :  0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float om = __shfl_xor_sync(0xffffffff, row_max, off);
            float os = __shfl_xor_sync(0xffffffff, row_sum, off);
            float nm = fmaxf(row_max, om);
            row_sum  = row_sum * expf(row_max - nm) + os * expf(om - nm);
            row_max  = nm;
        }
        if (lane_id == 0) { wmax_buf[0] = row_max; wsum_buf[0] = row_sum; }
    }
    __syncthreads();

    row_max = wmax_buf[0];
    float inv_sum = 1.0f / wsum_buf[0];

    // --- Phase 2: Normalize from shared memory, write float4 to DRAM ---
    for (int i = tid; i < D4; i += blockDim.x) {
        float4 v = scache4[i];
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
    // row_cache (D floats) + wmax (num_warps) + wsum (num_warps)
    int shmem     = (D + 2 * num_warps) * sizeof(float);
    softmax_v3<<<N, threads, shmem>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
