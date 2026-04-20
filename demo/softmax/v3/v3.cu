#include <cuda_runtime.h>
#include <cfloat>

// v3: Online softmax (Milakov & Gimelshein 2018) - 2-pass instead of 3-pass.
// Pass 1: Single forward scan over input computes (max, sum) simultaneously using
//         the running-max rescaling trick — avoids the separate max pass.
//         Result: no intermediate exp values written to output.
// Pass 2: Re-read input (hits L1 cache), write final normalized output.
// + float4 vectorized loads/stores throughout
// + __expf for fast transcendentals
// + #pragma unroll for inner loops

__global__ __launch_bounds__(256, 6)
void softmax_v3(const float* __restrict__ input, float* __restrict__ output,
                int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float4* in4  = reinterpret_cast<const float4*>(input  + (long)row * D);
    float4*       out4 = reinterpret_cast<float4*>      (output + (long)row * D);

    int D4 = D >> 2;

    // ---- pass 1: online max + sum (no intermediate store) ----
    float m = -FLT_MAX, d = 0.0f;

    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 v = in4[i];
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float m_new = fmaxf(m, vals[j]);
            d = d * __expf(m - m_new) + __expf(vals[j] - m_new);
            m = m_new;
        }
    }

    // warp-level reduce (m, d) pair
    for (int off = 16; off > 0; off >>= 1) {
        float m_o = __shfl_down_sync(0xffffffff, m, off);
        float d_o = __shfl_down_sync(0xffffffff, d, off);
        float m_new = fmaxf(m, m_o);
        d = d * __expf(m - m_new) + d_o * __expf(m_o - m_new);
        m = m_new;
    }

    __shared__ float sm[32], sd[32];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int n_warps = blockDim.x >> 5;

    if (lid == 0) { sm[wid] = m; sd[wid] = d; }
    __syncthreads();

    if (wid == 0) {
        m = (lid < n_warps) ? sm[lid] : -FLT_MAX;
        d = (lid < n_warps) ? sd[lid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float m_o = __shfl_down_sync(0xffffffff, m, off);
            float d_o = __shfl_down_sync(0xffffffff, d, off);
            float m_new = fmaxf(m, m_o);
            d = d * __expf(m - m_new) + d_o * __expf(m_o - m_new);
            m = m_new;
        }
        if (lid == 0) { sm[0] = m; sd[0] = d; }
    }
    __syncthreads();
    m = sm[0];
    float inv_d = __fdividef(1.0f, sd[0]);

    // ---- pass 2: re-read input (L1 hot), write normalized output ----
    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 v = in4[i];
        float4 r;
        r.x = __expf(v.x - m) * inv_d;
        r.y = __expf(v.y - m) * inv_d;
        r.z = __expf(v.z - m) * inv_d;
        r.w = __expf(v.w - m) * inv_d;
        out4[i] = r;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    softmax_v3<<<N, 256>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
