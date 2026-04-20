#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64    // block tile rows
#define BN 64    // block tile cols
#define BK 16    // block tile K (== WMMA_K)

// 4 warps per block in 2×2 layout; each warp computes 2×2 WMMA tiles (32×32)
#define WARP_M 32
#define WARP_N 32

__global__ void wmma_gemm(
    const float* __restrict__ A_fp32,
    const float* __restrict__ B_fp32,
          float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ half As[BM][BK];   // 64×16 = 2 KB
    __shared__ half Bs[BK][BN];   // 16×64 = 2 KB

    const int tid     = threadIdx.x;            // 0..127
    const int warpId  = tid / 32;               // 0..3
    const int warpRow = warpId / 2;             // 0..1  (M direction)
    const int warpCol = warpId % 2;             // 0..1  (N direction)

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Accumulator fragments: 2×2 WMMA tiles per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    #pragma unroll
    for (int wi = 0; wi < 2; wi++)
        for (int wj = 0; wj < 2; wj++)
            wmma::fill_fragment(acc[wi][wj], 0.0f);

    for (int bk = 0; bk < K; bk += BK) {
        // Cooperative load A tile [BM×BK]: 128 threads × 8 fp32→fp16 each
        #pragma unroll
        for (int i = 0; i < BM * BK / 128; i++) {
            int idx = tid + i * 128;
            int r = idx / BK, c = idx % BK;
            int gr = blockRow + r, gc = bk + c;
            As[r][c] = (gr < M && gc < K) ? __float2half(A_fp32[gr * K + gc])
                                           : __float2half(0.0f);
        }
        // Cooperative load B tile [BK×BN]: 128 threads × 8 fp32→fp16 each
        #pragma unroll
        for (int i = 0; i < BK * BN / 128; i++) {
            int idx = tid + i * 128;
            int r = idx / BN, c = idx % BN;
            int gr = bk + r, gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? __float2half(B_fp32[gr * N + gc])
                                           : __float2half(0.0f);
        }

        __syncthreads();

        // Each warp computes its 2×2 block of WMMA tiles
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        #pragma unroll
        for (int wi = 0; wi < 2; wi++) {
            int rowOff = warpRow * WARP_M + wi * WMMA_M;
            wmma::load_matrix_sync(a_frag, &As[rowOff][0], BK);
            #pragma unroll
            for (int wj = 0; wj < 2; wj++) {
                int colOff = warpCol * WARP_N + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag, &Bs[0][colOff], BN);
                wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
            }
        }

        __syncthreads();
    }

    // Store 2×2 result fragments to global memory
    #pragma unroll
    for (int wi = 0; wi < 2; wi++) {
        #pragma unroll
        for (int wj = 0; wj < 2; wj++) {
            int row = blockRow + warpRow * WARP_M + wi * WMMA_M;
            int col = blockCol + warpCol * WARP_N + wj * WMMA_N;
            if (row < M && col < N)
                wmma::store_matrix_sync(&C[row * N + col], acc[wi][wj], N,
                                        wmma::mem_row_major);
        }
    }
}

extern "C" void solve(float* A, float* B, float* C, int M, int K, int N)
{
    dim3 threadsPerBlock(128);   // 4 warps
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    wmma_gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
