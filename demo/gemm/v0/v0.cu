#include <cuda_runtime.h>

__global__ void naive_gemm(
    const float* __restrict__ A, 
    const float* __restrict__ B,  
          float* __restrict__ C,  
    int M, int K, int N)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = acc;
}

extern "C" void solve(
    float* A, float* B, float* C,
    int M, int K, int N)
{

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    naive_gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}