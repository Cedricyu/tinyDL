#include "Linear.cuh"
#include "Tensor.cuh"
#include "CudaKernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

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

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto output = std::make_shared<Tensor>(input->batch_size, out_features);

    float *d_input;
    float *d_output;
    cudaMalloc(&d_input, input->size() * sizeof(float));
    cudaMalloc(&d_output, input->batch_size * out_features * sizeof(float));

    cudaMemcpy(d_input, input->data, input->size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (input->batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(
        d_input, d_weight, d_output, input->batch_size, in_features, out_features);
    addBiasKernel<<<gridSize, blockSize>>>(d_output, d_bias, input->batch_size, out_features);

    cudaMemcpy(output->data,
               d_output,
               input->batch_size * out_features * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

std::shared_ptr<Tensor> Linear::backward(std::shared_ptr<Tensor> input,
                                         std::shared_ptr<Tensor> grad_output) {
    float *d_input, *d_grad_output, *d_grad_input;

    cudaMalloc(&d_input, input->size() * sizeof(float));
    cudaMalloc(&d_grad_output, grad_output->size() * sizeof(float));
    cudaMalloc(&d_grad_input, input->size() * sizeof(float));

    cudaMemcpy(d_input, input->data, input->size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output,
               grad_output->data,
               grad_output->size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // grad_weight
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSizeW((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridSizeW, blockSize>>>(
        d_grad_output, d_input, d_grad_weight, out_features, input->batch_size, in_features);

    // grad_bias
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    biasGradientKernel<<<blocks, threads>>>(
        d_grad_output, d_grad_bias, input->batch_size, out_features);

    // grad_input
    dim3 gridSizeInput((in_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (input->batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridSizeInput, blockSize>>>(
        d_grad_output, d_weight, d_grad_input, input->batch_size, out_features, in_features);

    auto grad_input = std::make_shared<Tensor>(input->batch_size, in_features);
    cudaMemcpy(
        grad_input->data, d_grad_input, grad_input->size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);

    return grad_input;
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
