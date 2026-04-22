#include <cuda_runtime.h>

// Block tile: BM rows × BN cols
// K tile:     BK
// Thread tile: TM × TN (each thread computes TM×TN outputs)
// Block size: (BN/TN) × (BM/TM) = 16 × 16 = 256 threads

#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

__global__ void tiled_gemm_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    // sA[BK][BM]: load A tile (transposed col-major in shared mem for reuse)
    // sB[BK][BN]: load B tile
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    int tx  = threadIdx.x;        // 0..BN/TN-1 = 0..15
    int ty  = threadIdx.y;        // 0..BM/TM-1 = 0..15
    int tid = ty * (BN / TN) + tx; // 0..255

    int baseRow = blockIdx.y * BM;
    int baseCol = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // BM*BK = 512 elements, 256 threads → 2 each; same for B
    const int A_LOADS = BM * BK / 256;   // = 2
    const int B_LOADS = BK * BN / 256;   // = 2

    for (int t = 0; t < K; t += BK) {

        // Load A tile: sA[k][m] = A[baseRow+m][t+k]
        // Pattern: consecutive threads pick consecutive k offsets → coalesced within row
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            int flat  = tid + i * 256;
            int k_off = flat % BK;      // 0..7 (K direction, consecutive per group of BK)
            int m_off = flat / BK;      // 0..63 (M direction)
            int gRow  = baseRow + m_off;
            int gCol  = t + k_off;
            sA[k_off][m_off] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        // Load B tile: sB[k][n] = B[t+k][baseCol+n]
        // Pattern: consecutive threads pick consecutive n offsets → fully coalesced
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            int flat  = tid + i * 256;
            int k_off = flat / BN;      // 0..7
            int n_off = flat % BN;      // 0..63
            int gRow  = t + k_off;
            int gCol  = baseCol + n_off;
            sB[k_off][n_off] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // Accumulate TM×TN dot product from this tile
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

    // Write TM×TN outputs
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = baseRow + ty * TM + m;
        if (row >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int col = baseCol + tx * TN + n;
            if (col < N)
                C[row * N + col] = acc[m][n];
        }
    }
}

extern "C" void solve(
    float* A, float* B, float* C,
    int M, int K, int N)
{
    dim3 threads(BN / TN, BM / TM);   // (16, 16) = 256
    dim3 blocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    tiled_gemm_v2<<<blocks, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
