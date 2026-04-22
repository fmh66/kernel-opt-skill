#include <cuda_runtime.h>
#include <cfloat>

__global__ void naive_softmax(float* input, float* output, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float* in_row  = input  + row * D;
    float* out_row = output + row * D;

    float max_val = -FLT_MAX;
    for (int i = 0; i < D; i++) {
        max_val = fmaxf(max_val, in_row[i]);
    }

    float sum_val = 0.0f;
    for (int i = 0; i < D; i++) {
        float val = expf(in_row[i] - max_val);
        out_row[i] = val;
        sum_val += val;
    }

    for (int i = 0; i < D; i++) {
        out_row[i] /= sum_val;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    naive_softmax<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, D);
    cudaDeviceSynchronize();
}