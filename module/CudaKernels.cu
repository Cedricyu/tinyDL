#include "CudaKernels.cuh"

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + tx < N)
            shared_A[ty][tx] = A[row * N + t * BLOCK_SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if (col < K && t * BLOCK_SIZE + ty < N)
            shared_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * K + col];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += shared_A[ty][i] * shared_B[i][tx];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

__global__ void addBiasKernel(float *C, float *bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] += bias[col];
    }
}

__global__ void
biasGradientKernel(float *grad_output, float *grad_bias, int batch_size, int out_features) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= out_features)
        return;

    float sum = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        sum += grad_output[i * out_features + idx];
    }

    grad_bias[idx] = sum;
}

__global__ void matrixTransposeMulKernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // over N (Aᵗ 的 row)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // over K (B 的 col)

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            float a = A[i * N + row]; // Aᵗ[row, i] = A[i, row]
            float b = B[i * K + col]; // B[i, col]
            sum += a * b;
        }
        C[row * K + col] = sum;
    }
}
