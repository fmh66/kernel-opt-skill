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

// v5: Flash Attention (v4) + BLOCKQ=2.
//
// Process 2 query positions per block.  The K/V tile is shared between both
// groups → amortises the dk² smem load cost across 2 queries and doubles
// the number of resident warps per SM (4 vs 2), pushing occupancy from ~21%
// to ~42% and giving the scheduler more latency-hiding choices.
//
// Thread layout  (128 threads = 2 × dk):
//   group = t / dk        (0 or 1)
//   lt    = t % dk        (local dimension index 0..dk-1)
//   qi    = blockIdx.y*2 + group
//
// smem: [q_sm: 2*dk | kv_sm: dk*(dk+1) | tile_scores: 2*dk | warp_buf: 32]
// smem size = (2*dk + dk*(dk+1) + 2*dk + 32) * 4 = 17792 bytes (dk=64)
//   → 5 blocks/SM (same smem limit as v4) but 4 warps/block → 41.7% occupancy
//
// K/V loading is split between groups: group 0 loads rows 0..dk/2-1,
// group 1 loads rows dk/2..dk-1, halving each group's load-loop iterations.
__global__ void __launch_bounds__(128)
mha_v5_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N, int d_model, int h, int d_k)
{
    int head  = blockIdx.x;
    int bi    = blockIdx.y;          // block index over pairs of queries
    int t     = threadIdx.x;         // 0 .. 2*dk-1
    int group = t / d_k;             // 0 or 1
    int lt    = t % d_k;             // local thread 0..dk-1
    int qi    = bi * 2 + group;      // actual query position

    int warp_id         = t >> 5;    // 0..3
    int lane            = t & 31;
    int warps_per_group = d_k >> 5;  // 2 for dk=64

    extern __shared__ float smem[];
    float* q_sm        = smem;
    float* kv_sm       = q_sm + 2 * d_k;               // [dk*(dk+1)]
    float* tile_scores = kv_sm + d_k * (d_k + 1);      // [2*dk]
    float* warp_buf    = tile_scores + 2 * d_k;         // [32]

    // Load Q: thread t → q_sm[t] = Q[qi * d_model + head*dk + lt]
    q_sm[t] = Q[qi * d_model + head * d_k + lt];
    __syncthreads();

    float scale = rsqrtf((float)d_k);
    float m_t   = -FLT_MAX;
    float l_t   = 0.0f;
    float o_t   = 0.0f;

    for (int j_start = 0; j_start < N; j_start += d_k) {
        // ── Load K tile cooperatively ────────────────────────────────────
        // Each group loads half the rows → dk/2 loop iters per group.
        // Coalesced: thread lt loads column lt for each row.
        int row_lo = group * (d_k >> 1);
        int row_hi = row_lo + (d_k >> 1);
        for (int row = row_lo; row < row_hi; row++) {
            kv_sm[row * (d_k + 1) + lt] = K[(j_start + row) * d_model + head * d_k + lt];
        }
        __syncthreads();

        // ── Dot product Q[qi] · K[j_start + lt] ─────────────────────────
        // q_sm[group*dk+d]: broadcast to all threads in group → free.
        // kv_sm[lt*(dk+1)+d]: bank = (lt+d)%32, conflict-free across warps.
        float a0=0, a1=0, a2=0, a3=0;
        #pragma unroll 4
        for (int d = 0; d < d_k; d += 4) {
            a0 = __fmaf_rn(q_sm[group*d_k+d+0], kv_sm[lt*(d_k+1)+(d+0)], a0);
            a1 = __fmaf_rn(q_sm[group*d_k+d+1], kv_sm[lt*(d_k+1)+(d+1)], a1);
            a2 = __fmaf_rn(q_sm[group*d_k+d+2], kv_sm[lt*(d_k+1)+(d+2)], a2);
            a3 = __fmaf_rn(q_sm[group*d_k+d+3], kv_sm[lt*(d_k+1)+(d+3)], a3);
        }
        float raw_score = (a0 + a1 + a2 + a3) * scale;

        // ── Per-group block-reduce: tile max ─────────────────────────────
        float wmax = warpReduceMax(raw_score);
        if (lane == 0) warp_buf[warp_id] = wmax;
        __syncthreads();
        if (lt == 0) {          // t==0 for group 0, t==dk for group 1
            int base = group * warps_per_group;
            float gmax = warp_buf[base];
            for (int w = 1; w < warps_per_group; w++) gmax = fmaxf(gmax, warp_buf[base + w]);
            warp_buf[base] = gmax;
        }
        __syncthreads();
        float tile_max = warp_buf[group * warps_per_group];
        // Extra sync: all warps must read tile_max before any warp overwrites
        // warp_buf[group*wpg] for the sum reduction (same slot).
        __syncthreads();

        // ── Online softmax update ────────────────────────────────────────
        float m_new   = fmaxf(m_t, tile_max);
        float rescale = expf(m_t - m_new);
        float w_t     = expf(raw_score - m_new);
        o_t  *= rescale;

        // ── Per-group block-reduce: tile sum ─────────────────────────────
        float wsum = warpReduceSum(w_t);
        if (lane == 0) warp_buf[warp_id] = wsum;
        tile_scores[group * d_k + lt] = w_t;
        __syncthreads();
        if (lt == 0) {
            int base = group * warps_per_group;
            float gs = warp_buf[base];
            for (int w = 1; w < warps_per_group; w++) gs += warp_buf[base + w];
            warp_buf[base] = gs;
        }
        __syncthreads();
        l_t = l_t * rescale + warp_buf[group * warps_per_group];
        m_t = m_new;

        // ── Load V tile cooperatively (reuse kv_sm) ──────────────────────
        for (int row = row_lo; row < row_hi; row++) {
            kv_sm[row * (d_k + 1) + lt] = V[(j_start + row) * d_model + head * d_k + lt];
        }
        __syncthreads();

        // ── Accumulate output (4-way ILP) ────────────────────────────────
        // tile_scores[group*dk+jt]: broadcast within group → free.
        // kv_sm[jt*(dk+1)+lt]: bank=(jt+lt)%32, conflict-free.
        float v0=0, v1=0, v2=0, v3=0;
        int ts_base = group * d_k;
        #pragma unroll 4
        for (int jt = 0; jt < d_k; jt += 4) {
            v0 = __fmaf_rn(tile_scores[ts_base+jt+0], kv_sm[(jt+0)*(d_k+1)+lt], v0);
            v1 = __fmaf_rn(tile_scores[ts_base+jt+1], kv_sm[(jt+1)*(d_k+1)+lt], v1);
            v2 = __fmaf_rn(tile_scores[ts_base+jt+2], kv_sm[(jt+2)*(d_k+1)+lt], v2);
            v3 = __fmaf_rn(tile_scores[ts_base+jt+3], kv_sm[(jt+3)*(d_k+1)+lt], v3);
        }
        o_t += v0 + v1 + v2 + v3;

        __syncthreads();    // protect kv_sm before next K load
    }

    output[qi * d_model + head * d_k + lt] = o_t / l_t;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;
    dim3 grid(num_heads, N / 2);
    dim3 block(2 * d_k);
    // smem: q_sm[2*dk] + kv_sm[dk*(dk+1)] + tile_scores[2*dk] + warp_buf[32]
    size_t smem = (size_t)(2*d_k + d_k*(d_k+1) + 2*d_k + 32) * sizeof(float);
    mha_v5_kernel<<<grid, block, smem>>>(Q, K, V, output, N, d_model, num_heads, d_k);
    cudaDeviceSynchronize();
}
