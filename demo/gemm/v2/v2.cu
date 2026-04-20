#include <cuda_runtime.h>

// Block tile: each block computes BM×BN of C
// Thread tile: each thread computes TM×TN output elements
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
// blockDim = (BN/TN, BM/TM) = (16, 16) = 256 threads

__global__ void regblock_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float As[BM][BK];   // 64×16 = 4 KB
    __shared__ float Bs[BK][BN];   // 16×64 = 4 KB

    const int tx  = threadIdx.x;   // 0..15
    const int ty  = threadIdx.y;   // 0..15
    const int tid = ty * blockDim.x + tx;   // 0..255

    const int tileRow = blockIdx.y * BM;
    const int tileCol = blockIdx.x * BN;

    float acc[TM][TN] = {};

    for (int bk = 0; bk < K; bk += BK) {
        // Cooperative load of As[BM][BK]: 256 threads × 4 elements each
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int r = idx / BK, c = idx % BK;
            int gr = tileRow + r, gc = bk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // Cooperative load of Bs[BK][BN]: 256 threads × 4 elements each
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int r = idx / BN, c = idx % BN;
            int gr = bk + r, gc = tileCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        float regA[TM], regB[TN];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) regA[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) regB[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // Write back TM×TN results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = tileRow + ty * TM + i;
            int col = tileCol + tx * TN + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}

extern "C" void solve(float* A, float* B, float* C, int M, int K, int N)
{
    dim3 threadsPerBlock(BN / TN, BM / TM);   // (16, 16)
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    regblock_gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
