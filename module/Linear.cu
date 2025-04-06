#include "Linear.cuh"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

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

Linear::Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
    cudaMalloc(&d_weight, in_f * out_f * sizeof(float));
    cudaMalloc(&d_bias, out_f * sizeof(float));
    cudaMalloc(&d_grad_weight, in_f * out_f * sizeof(float));
    cudaMalloc(&d_grad_bias, out_f * sizeof(float));

    float *h_weight = (float *)malloc(in_f * out_f * sizeof(float));
    float *h_bias = (float *)malloc(out_f * sizeof(float));

    float std = sqrtf(2.0f / in_f);
    for (int i = 0; i < in_f * out_f; ++i)
        h_weight[i] = std * ((rand() / float(RAND_MAX)) * 2 - 1);

    for (int i = 0; i < out_f; ++i)
        h_bias[i] = 0.0f;

    cudaMemcpy(d_weight, h_weight, in_f * out_f * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, out_f * sizeof(float), cudaMemcpyHostToDevice);

    free(h_weight);
    free(h_bias);
}

void Linear::forward(float *input, float *output, int batch_size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_features * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(
        d_input, d_weight, d_output, batch_size, in_features, out_features);
    addBiasKernel<<<gridSize, blockSize>>>(d_output, d_bias, batch_size, out_features);

    cudaMemcpy(output, d_output, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

std::vector<float> Linear::backward(float *input, float *grad_output, int batch_size) {
    float *d_input, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * out_features * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * in_features * sizeof(float)); // ⬅️ NEW

    cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output,
               grad_output,
               batch_size * out_features * sizeof(float),
               cudaMemcpyHostToDevice);

    // 1. 計算權重梯度
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSizeW((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridSizeW, blockSize>>>(
        d_grad_output, d_input, d_grad_weight, out_features, batch_size, in_features);

    // 2. 計算 bias 梯度
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    biasGradientKernel<<<blocks, threads>>>(d_grad_output, d_grad_bias, batch_size, out_features);

    // 3. 計算 ∂L/∂input = grad_output × Wᵀ
    dim3 gridSizeInput((in_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridSizeInput, blockSize>>>(d_grad_output,
                                                       d_weight,
                                                       d_grad_input,
                                                       batch_size,
                                                       out_features,
                                                       in_features); // W must be transposed

    // 4. 複製結果回 host
    std::vector<float> h_grad_input(batch_size * in_features);
    cudaMemcpy(h_grad_input.data(),
               d_grad_input,
               batch_size * in_features * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);

    return h_grad_input;
}

Linear::~Linear() {
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_grad_weight);
    cudaFree(d_grad_bias);
}

float Linear::get_weight(int in_idx, int out_idx) const {
    float value;
    // 權重在 GPU 上是以 row-major 排列: [out_features][in_features]
    // 所以取值 index 為: out_idx * in_features + in_idx
    cudaMemcpy(
        &value, d_weight + out_idx * in_features + in_idx, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

void Linear::print_weight(const std::string &name) const {
    std::vector<float> h_weight(in_features * out_features);
    std::vector<float> h_bias(out_features);
    std::vector<float> h_grad_weight(in_features * out_features);
    std::vector<float> h_grad_bias(out_features);

    cudaMemcpy(h_weight.data(),
               d_weight,
               sizeof(float) * in_features * out_features,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bias.data(), d_bias, sizeof(float) * out_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_weight.data(),
               d_grad_weight,
               sizeof(float) * in_features * out_features,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(
        h_grad_bias.data(), d_grad_bias, sizeof(float) * out_features, cudaMemcpyDeviceToHost);

    std::cout << "[Linear Layer] " << name << " Weights (partial): ";
    for (int i = 0; i < std::min(10, in_features * out_features); ++i) {
        std::cout << h_weight[i] << " ";
    }
    std::cout << "\n";

    std::cout << "[Linear Layer] " << name << " Weight Gradients (partial): ";
    for (int i = 0; i < std::min(10, in_features * out_features); ++i) {
        std::cout << h_grad_weight[i] << " ";
    }
    std::cout << "\n";

    std::cout << "[Linear Layer] " << name << " Bias (partial): ";
    for (int i = 0; i < std::min(10, out_features); ++i) {
        std::cout << h_bias[i] << " ";
    }
    std::cout << "\n";

    std::cout << "[Linear Layer] " << name << " Bias Gradients (partial): ";
    for (int i = 0; i < std::min(10, out_features); ++i) {
        std::cout << h_grad_bias[i] << " ";
    }
    std::cout << "\n";
}
