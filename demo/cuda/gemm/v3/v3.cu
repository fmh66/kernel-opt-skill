#include <cuda_runtime.h>

// Block tile: BM × BN, K tile: BK
// Thread tile: TM × TN (each thread owns TM×TN accumulators)
// Block: (BN/TN) × (BM/TM) = 16 × 8 = 128 threads
//
// BK=16: 1 float4 load per thread per tile (BM*BK/128=8 floats=2 float4, BK*BN/128=8 floats=2 float4)
// Padding on sA columns eliminates 2-way bank conflict from stride-TM access.

#define BM  64
#define BN  64
#define BK  16
#define TM   8
#define TN   4

// Padding: sA[BK][BM+1] — odd stride breaks bank-conflict pattern for stride-TM reads
#define SMA_PAD 1

__global__ void tiled_gemm_v3(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float sA[BK][BM + SMA_PAD];
    __shared__ float sB[BK][BN];

    const int tx  = threadIdx.x;                  // 0..BN/TN-1 = 0..15
    const int ty  = threadIdx.y;                  // 0..BM/TM-1 = 0..7
    const int tid = ty * (BN / TN) + tx;          // 0..127

    const int baseRow = blockIdx.y * BM;
    const int baseCol = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // Block has BLOCK_SIZE = (BM/TM)*(BN/TN) = 8*16 = 128 threads
    // A tile = BM*BK = 1024 elements → 8 floats per thread
    // B tile = BK*BN = 1024 elements → 8 floats per thread
    constexpr int BLOCK_SIZE = (BM / TM) * (BN / TN);   // 128
    constexpr int A_LOADS    = BM * BK / BLOCK_SIZE;     // 8
    constexpr int B_LOADS    = BK * BN / BLOCK_SIZE;     // 8

    for (int t = 0; t < K; t += BK) {

        // --- Load A tile: sA[k][m] = A[baseRow+m][t+k] ---
        // k_off = flat % BK → groups of BK consecutive threads share same M row, coalesced K access
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            int flat  = tid + i * BLOCK_SIZE;
            int k_off = flat % BK;
            int m_off = flat / BK;
            int gRow  = baseRow + m_off;
            int gCol  = t + k_off;
            sA[k_off][m_off] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        // --- Load B tile: sB[k][n] = B[t+k][baseCol+n] ---
        // n_off = flat % BN → BN consecutive threads access consecutive N columns → coalesced
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            int flat  = tid + i * BLOCK_SIZE;
            int k_off = flat / BN;
            int n_off = flat % BN;
            int gRow  = t + k_off;
            int gCol  = baseCol + n_off;
            sB[k_off][n_off] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // --- Compute TM×TN tile ---
        float regA[TM], regB[TN];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++)
                regA[m] = sA[k][ty * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                regB[n] = sB[k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // --- Write TM×TN outputs with vectorized float4 stores ---
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = baseRow + ty * TM + m;
        if (row >= M) continue;
        // Store TN=4 consecutive columns as float4
        int col = baseCol + tx * TN;
        if (col + TN - 1 < N) {
            float4 out = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            *reinterpret_cast<float4*>(&C[row * N + col]) = out;
        } else {
            #pragma unroll
            for (int n = 0; n < TN; n++)
                if (col + n < N)
                    C[row * N + col + n] = acc[m][n];
        }
    }
}

extern "C" void solve(
    float* A, float* B, float* C,
    int M, int K, int N)
{
    dim3 threads(BN / TN, BM / TM);   // (16, 8) = 128
    dim3 blocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    tiled_gemm_v3<<<blocks, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
