#include "CudaKernels.cuh"
#include "Linear.cuh"
#include "Tensor.cuh"
#include "TensorOps.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>

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

// std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
//     int batch = input->batch_size;
//     int in_feat = this->in_features;
//     int out_feat = this->out_features;

//     // 呼叫 matmul kernel 並返回 CPU tensor
//     Tensor *raw_output = tensor_matmul(input.get(), NULL, d_weight, d_bias);  // NULL: b 是 GPU 常數權重，不是 Tensor

//     std::shared_ptr<Tensor> output(raw_output);

//     // 建立 backward dependency（若支援）
//     if (input->requires_grad) {
//         tensor_add_dependency(output.get(), input.get(), /*Linear backward fn (未定義)*/ NULL);
//     }

//     return output;
// }

// std::shared_ptr<Tensor> Linear::backward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> grad_output) {
//     Tensor *grad_input = tensor_matmul_backward(
//         input.get(), grad_output.get(),
//         d_weight, d_grad_weight, d_grad_bias,
//         in_features, out_features);

//     return std::shared_ptr<Tensor>(grad_input);
// }

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
    cudaMemcpy(&value, d_weight + out_idx * in_features + in_idx, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

void Linear::print_weight(const std::string &name) const {
    std::vector<float> h_weight(in_features * out_features);
    std::vector<float> h_bias(out_features);
    std::vector<float> h_grad_weight(in_features * out_features);
    std::vector<float> h_grad_bias(out_features);

    cudaMemcpy(h_weight.data(), d_weight, sizeof(float) * in_features * out_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bias.data(), d_bias, sizeof(float) * out_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_weight.data(), d_grad_weight, sizeof(float) * in_features * out_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_bias.data(), d_grad_bias, sizeof(float) * out_features, cudaMemcpyDeviceToHost);

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
