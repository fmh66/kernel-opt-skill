#include <cuda_runtime.h>

#define TILE 16

__global__ void tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int kA = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.0f;

        int kB = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (kB < K && col < N) ? B[kB * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

extern "C" void solve(float* A, float* B, float* C, int M, int K, int N)
{
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );
    tiled_gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
