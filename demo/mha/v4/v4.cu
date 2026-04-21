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

// v4: Flash Attention — online softmax with K/V tiling in shared memory.
//
// Key improvements over v2/v3:
//   1. Coalesced K/V loading: ALL threads load row-t, column varies per thread
//      → 2 cache lines per K row vs dk separate scattered loads in v2
//   2. K/V accessed from smem (20-cycle latency) instead of L2 (200-cycle)
//      → eliminates the dominant Long Scoreboard stall
//   3. Online softmax removes the N-element scores buffer from smem
//      → more smem available but we use it for k_tile
//   4. Bank-conflict-free smem with +1-float row padding
//
// Tradeoff: 5 blocks/SM vs 22 in v2 (lower occupancy), but memory access
// quality improvement dominates.
//
// smem: [q_sm: dk | kv_sm: dk*(dk+1) | tile_scores: dk | warp_buf: 32]
__global__ void __launch_bounds__(64)
mha_v4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N, int d_model, int h, int d_k)
{
    int head      = blockIdx.x;
    int i         = blockIdx.y;
    int t         = threadIdx.x;
    int num_warps = (d_k + 31) >> 5;
    int warp_id   = t >> 5;
    int lane      = t & 31;

    extern __shared__ float smem[];
    float* q_sm        = smem;
    float* kv_sm       = q_sm + d_k;                  // [dk * (dk+1)]
    float* tile_scores = kv_sm + d_k * (d_k + 1);     // [dk]
    float* warp_buf    = tile_scores + d_k;            // [32]

    // Load Q (coalesced, broadcast later from smem)
    q_sm[t] = Q[i * d_model + head * d_k + t];
    __syncthreads();

    float scale = rsqrtf((float)d_k);
    float m_t   = -FLT_MAX;  // running max (scalar, same for all threads)
    float l_t   = 0.0f;      // running sum of exp weights (scalar)
    float o_t   = 0.0f;      // running output for dimension t (per-thread)

    for (int j_start = 0; j_start < N; j_start += d_k) {
        // ── Load K tile (coalesced) ──────────────────────────────────────
        // Thread t loads kv_sm[row][t] = K[j_start+row, head*dk+t]
        // For each row: all dk threads load consecutive K columns → coalesced
        for (int row = 0; row < d_k; row++) {
            kv_sm[row * (d_k + 1) + t] = K[(j_start + row) * d_model + head * d_k + t];
        }
        __syncthreads();

        // ── Dot product Q·K_tile[t] (4-way ILP, no bank conflicts) ──────
        // kv_sm[t*(dk+1)+d]: thread t, varying d → banks (t+d)%32 ✓
        float a0=0, a1=0, a2=0, a3=0;
        #pragma unroll 4
        for (int d = 0; d < d_k; d += 4) {
            a0 = __fmaf_rn(q_sm[d+0], kv_sm[t*(d_k+1)+(d+0)], a0);
            a1 = __fmaf_rn(q_sm[d+1], kv_sm[t*(d_k+1)+(d+1)], a1);
            a2 = __fmaf_rn(q_sm[d+2], kv_sm[t*(d_k+1)+(d+2)], a2);
            a3 = __fmaf_rn(q_sm[d+3], kv_sm[t*(d_k+1)+(d+3)], a3);
        }
        float raw_score = (a0 + a1 + a2 + a3) * scale;

        // ── Block reduce: tile max ───────────────────────────────────────
        float wmax = warpReduceMax(raw_score);
        if (lane == 0) warp_buf[warp_id] = wmax;
        __syncthreads();
        if (t == 0) {
            float gmax = warp_buf[0];
            for (int w = 1; w < num_warps; w++) gmax = fmaxf(gmax, warp_buf[w]);
            warp_buf[0] = gmax;
        }
        __syncthreads();
        float tile_max = warp_buf[0];
        // All warps must capture tile_max before warp 0 overwrites warp_buf[0] for the sum.
        __syncthreads();

        // ── Online softmax update ────────────────────────────────────────
        float m_new   = fmaxf(m_t, tile_max);
        float rescale = expf(m_t - m_new);
        float w_t     = expf(raw_score - m_new);
        o_t  *= rescale;

        // Block reduce: tile sum (writes w_t to smem simultaneously)
        float wsum = warpReduceSum(w_t);
        if (lane == 0) warp_buf[warp_id] = wsum;
        tile_scores[t] = w_t;               // write while warp_buf fills
        __syncthreads();
        if (t == 0) {
            float gs = warp_buf[0];
            for (int w = 1; w < num_warps; w++) gs += warp_buf[w];
            warp_buf[0] = gs;
        }
        __syncthreads();
        l_t = l_t * rescale + warp_buf[0];
        m_t = m_new;

        // ── Load V tile (reuse kv_sm; tile_scores is unmodified) ────────
        for (int row = 0; row < d_k; row++) {
            kv_sm[row * (d_k + 1) + t] = V[(j_start + row) * d_model + head * d_k + t];
        }
        __syncthreads();  // V ready AND tile_scores visible to all warps

        // ── Accumulate output (4-way ILP) ────────────────────────────────
        // kv_sm[jt*(dk+1)+t]: bank (jt+t)%32, conflict-free ✓
        float v0=0, v1=0, v2=0, v3=0;
        #pragma unroll 4
        for (int jt = 0; jt < d_k; jt += 4) {
            v0 = __fmaf_rn(tile_scores[jt+0], kv_sm[(jt+0)*(d_k+1)+t], v0);
            v1 = __fmaf_rn(tile_scores[jt+1], kv_sm[(jt+1)*(d_k+1)+t], v1);
            v2 = __fmaf_rn(tile_scores[jt+2], kv_sm[(jt+2)*(d_k+1)+t], v2);
            v3 = __fmaf_rn(tile_scores[jt+3], kv_sm[(jt+3)*(d_k+1)+t], v3);
        }
        o_t += v0 + v1 + v2 + v3;

        __syncthreads();  // protect kv_sm before next K load
    }

    output[i * d_model + head * d_k + t] = o_t / l_t;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;
    dim3 grid(num_heads, N);
    dim3 block(d_k);
    // smem: q_sm[dk] + kv_sm[dk*(dk+1)] + tile_scores[dk] + warp_buf[32]
    size_t smem = (size_t)(d_k + d_k * (d_k + 1) + d_k + 32) * sizeof(float);
    mha_v4_kernel<<<grid, block, smem>>>(Q, K, V, output, N, d_model, num_heads, d_k);
    cudaDeviceSynchronize();
}
