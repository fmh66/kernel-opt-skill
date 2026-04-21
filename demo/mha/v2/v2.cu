#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

// v2: float4 loads + 4-way ILP accumulators to break FMA dependency chain.
// smem: [q4_sm: d_k floats | scores: N floats | warp_buf: 32 floats]
__global__ void __launch_bounds__(64, 16)
mha_v2_kernel(
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
    float4* q4_sm   = (float4*)smem;         // d_k floats as float4
    float*  scores  = smem + d_k;
    float*  warp_buf = smem + d_k + N;

    // Cooperatively load Q in float4 chunks (only dk/4 threads needed)
    int dk4 = d_k >> 2;
    if (t < dk4) {
        q4_sm[t] = __ldg((const float4*)(Q + i * d_model + head * d_k) + t);
    }
    __syncthreads();

    float scale = rsqrtf((float)d_k);

    // Each thread computes scores for j = t, t+dk, ...
    // Float4 loads + 4 independent accumulators to break FMA chain
    for (int j = t; j < N; j += d_k) {
        const float4* kptr4 = (const float4*)(K + j * d_model + head * d_k);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        #pragma unroll 4
        for (int d = 0; d < dk4; d++) {
            float4 k4 = __ldg(kptr4 + d);
            float4 q4 = q4_sm[d];
            a0 = __fmaf_rn(q4.x, k4.x, a0);
            a1 = __fmaf_rn(q4.y, k4.y, a1);
            a2 = __fmaf_rn(q4.z, k4.z, a2);
            a3 = __fmaf_rn(q4.w, k4.w, a3);
        }
        scores[j] = (a0 + a1 + a2 + a3) * scale;
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

    // Exp + sum
    float my_sum = 0.f;
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
    float inv_sum = __fdividef(1.f, warp_buf[0]);

    for (int j = t; j < N; j += d_k)
        scores[j] *= inv_sum;
    __syncthreads();

    // Output accumulation: 4-way unroll for ILP
    const float* vbase = V + head * d_k + t;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    int j = 0;
    for (; j + 3 < N; j += 4) {
        o0 = __fmaf_rn(scores[j+0], __ldg(vbase + (j+0)*d_model), o0);
        o1 = __fmaf_rn(scores[j+1], __ldg(vbase + (j+1)*d_model), o1);
        o2 = __fmaf_rn(scores[j+2], __ldg(vbase + (j+2)*d_model), o2);
        o3 = __fmaf_rn(scores[j+3], __ldg(vbase + (j+3)*d_model), o3);
    }
    for (; j < N; j++)
        o0 = __fmaf_rn(scores[j], __ldg(vbase + j*d_model), o0);
    output[i * d_model + head * d_k + t] = o0 + o1 + o2 + o3;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;
    dim3 grid(num_heads, N);
    dim3 block(d_k);
    size_t smem = (size_t)(d_k + N + 32) * sizeof(float);
    mha_v2_kernel<<<grid, block, smem>>>(Q, K, V, output, N, d_model, num_heads, d_k);
    cudaDeviceSynchronize();
}
