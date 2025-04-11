#include "ActivationKernels.cuh"
#include "CudaHelper.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

const int THREADS_PER_BLOCK = 256;
const int TEST_SIZE = 1024;
const float LEKEY_RELU_ALPHA = 0.01f;

// Generic Activation Kernel Setup ---
//   Note: A more robust solution might use std::function or templates if complexity increases.
typedef void (*HostForwardFunc)(float *h_input, float *h_output, float param, int size);
typedef void (*HostBackwardFunc)(float *h_input, float *h_grad_output, float *h_grad_input, float param, int size);

void test_activation_kernel(const std::string &name,
                            HostForwardFunc forward_func,
                            HostBackwardFunc backward_func,
                            float param,
                            int size,
                            int count = THREADS_PER_BLOCK) {
    std::cout << name << " Check: " << std::endl;

    float *h_input = new float[size];
    float *h_output = new float[size];
    float *h_grad_input = new float[size];
    float *h_grad_output = new float[size];

    for (int i = 0; i < size; ++i) {
        h_input[i] = -2.0f + (4.0f * i) / (size > 1 ? (size - 1) : 1);
        h_grad_input[i] = 0.0f;
        h_grad_output[i] = 1.0f;
    }

    forward_func(h_input, h_output, param, size);
    backward_func(h_input, h_grad_output, h_grad_input, param, size);

    count = std::min(count, size);
    for (int i = 0; i < count; ++i) {
        std::cout << "Input: " << h_input[i] << ", Foward Output: " << h_output[i]
                  << ", Grad Output: " << h_grad_output[i] << ", Grad Input: " << h_grad_input[i] << std::endl;
    }

    delete[] h_input;
    delete[] h_output;
    delete[] h_grad_input;
    delete[] h_grad_output;
    std::cout << name << " Check Done!" << std::endl;
}

void sigmoid_forward(float *h_input, float *h_output, int size) {
    float *d_input, *d_output;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sigmoidKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_output, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_output to host.");

    cudaFree(d_input);
    cudaFree(d_output);
}

void sigmoid_forward_wrapper(float *h_input, float *h_output, float _, int size) {
    sigmoid_forward(h_input, h_output, size);
}

void sigmoid_backward(float *h_input, float *h_grad_output, float *h_grad_input, int size) {
    float *d_input, *d_grad_output, *d_grad_input;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_grad_output, size * sizeof(float)), "Failed to allocate d_grad_output.");
    checkCudaError(cudaMalloc(&d_grad_input, size * sizeof(float)), "Failed to allocate d_grad_input.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");
    checkCudaError(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_grad_output to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sigmoidBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_grad_output, d_grad_input, size);
    checkCudaError(cudaGetLastError(), "Backward Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_grad_input to host.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void sigmoid_backward_wrapper(float *h_input, float *h_grad_output, float *h_grad_input, float _, int size) {
    sigmoid_backward(h_input, h_grad_output, h_grad_input, size);
}

void relu_forward(float *h_input, float *h_output, int size) {
    float *d_input, *d_output;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    reluKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_output, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_output to host.");

    cudaFree(d_input);
    cudaFree(d_output);
}

void relu_forward_wrapper(float *h_input, float *h_output, float _, int size) {
    relu_forward(h_input, h_output, size);
}

void relu_backward(float *h_input, float *h_grad_output, float *h_grad_input, int size) {
    float *d_input, *d_grad_output, *d_grad_input;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_grad_output, size * sizeof(float)), "Failed to allocate d_grad_output.");
    checkCudaError(cudaMalloc(&d_grad_input, size * sizeof(float)), "Failed to allocate d_grad_input.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");
    checkCudaError(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_grad_output to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    reluBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_grad_output, d_grad_input, size);
    checkCudaError(cudaGetLastError(), "Backward Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_grad_input to host.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void relu_backward_wrapper(float *h_input, float *h_grad_output, float *h_grad_input, float _, int size) {
    relu_backward(h_input, h_grad_output, h_grad_input, size);
}

void leaky_relu_forward(float *h_input, float *h_output, float alpha, int size) {
    float *d_input, *d_output;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");

    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    leakyReluKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_output, alpha, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_output to host.");

    cudaFree(d_input);
    cudaFree(d_output);
}

void leaky_relu_backward(float *h_input, float *h_grad_output, float *h_grad_input, float alpha, int size) {
    float *d_input, *d_grad_output, *d_grad_input;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_grad_output, size * sizeof(float)), "Failed to allocate d_grad_output.");
    checkCudaError(cudaMalloc(&d_grad_input, size * sizeof(float)), "Failed to allocate d_grad_input.");

    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");
    checkCudaError(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_grad_output to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    leakyReluBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_grad_output, d_grad_input, alpha, size);
    checkCudaError(cudaGetLastError(), "Backward Kernel launch failed.");

    checkCudaError(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_grad_input to host.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void elu_forward(float *h_input, float *h_output, float alpha, int size) {
    float *d_input, *d_output;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    eluKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_output, alpha, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_output to host.");

    cudaFree(d_input);
    cudaFree(d_output);
}

void elu_backward(float *h_input, float *h_grad_output, float *h_grad_input, float alpha, int size) {
    float *d_input, *d_grad_output, *d_grad_input;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_grad_output, size * sizeof(float)), "Failed to allocate d_grad_output.");
    checkCudaError(cudaMalloc(&d_grad_input, size * sizeof(float)), "Failed to allocate d_grad_input.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");
    checkCudaError(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_grad_output to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    eluBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_grad_output, d_grad_input, alpha, size);
    checkCudaError(cudaGetLastError(), "Backward Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_grad_input to host.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void softmax_forward(float *h_input, float *h_output, int size) {
    float *d_input, *d_output;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    softmaxKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_output, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_output to host.");

    cudaFree(d_input);
    cudaFree(d_output);
}

void softmax_forward_wrapper(float *h_input, float *h_output, float _, int size) {
    softmax_forward(h_input, h_output, size);
}

void softmax_backward(float *h_input, float *h_grad_output, float *h_grad_input, int size) {
    float *d_input, *d_grad_output, *d_grad_input;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_grad_output, size * sizeof(float)), "Failed to allocate d_grad_output.");
    checkCudaError(cudaMalloc(&d_grad_input, size * sizeof(float)), "Failed to allocate d_grad_input.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");
    checkCudaError(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_grad_output to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    softmaxBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_grad_output, d_grad_input, size);
    checkCudaError(cudaGetLastError(), "Backward Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_grad_input to host.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void softmax_backward_wrapper(float *h_input, float *h_grad_output, float *h_grad_input, float _, int size) {
    softmax_backward(h_input, h_grad_output, h_grad_input, size);
}

void tanh_forward(float *h_input, float *h_output, int size) {
    float *d_input, *d_output;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tanhKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_output, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_output to host.");

    cudaFree(d_input);
    cudaFree(d_output);
}

void tanh_forward_wrapper(float *h_input, float *h_output, float _, int size) {
    tanh_forward(h_input, h_output, size);
}

void tanh_backward(float *h_input, float *h_grad_output, float *h_grad_input, int size) {
    float *d_input, *d_grad_output, *d_grad_input;

    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_grad_output, size * sizeof(float)), "Failed to allocate d_grad_output.");
    checkCudaError(cudaMalloc(&d_grad_input, size * sizeof(float)), "Failed to allocate d_grad_input.");
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");
    checkCudaError(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_grad_output to device.");

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tanhBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_grad_output, d_grad_input, size);
    checkCudaError(cudaGetLastError(), "Backward Kernel launch failed.");
    checkCudaError(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy d_grad_input to host.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void tanh_backward_wrapper(float *h_input, float *h_grad_output, float *h_grad_input, float _, int size) {
    tanh_backward(h_input, h_grad_output, h_grad_input, size);
}

int main() {
    std::cout << "Activation Kernel Checks Start." << std::endl;
    std::cout << "========================" << std::endl;

    test_activation_kernel("Sigmoid", sigmoid_forward_wrapper, sigmoid_backward_wrapper, 0.0f, TEST_SIZE);

    std::cout << "------------------------" << std::endl;

    test_activation_kernel("ReLU", relu_forward_wrapper, relu_backward_wrapper, 0.0f, TEST_SIZE);

    std::cout << "------------------------" << std::endl;

    test_activation_kernel("Leaky ReLU", leaky_relu_forward, leaky_relu_backward, LEKEY_RELU_ALPHA, TEST_SIZE);

    std::cout << "------------------------" << std::endl;

    test_activation_kernel("ELU", elu_forward, elu_backward, 1.0f, TEST_SIZE);

    std::cout << "------------------------" << std::endl;

    test_activation_kernel("Softmax", softmax_forward_wrapper, softmax_backward_wrapper, 0.0f, TEST_SIZE);

    std::cout << "------------------------" << std::endl;

    test_activation_kernel("Tanh", tanh_forward_wrapper, tanh_backward_wrapper, 0.0f, TEST_SIZE);

    std::cout << "------------------------" << std::endl;

    std::cout << "========================" << std::endl;
    std::cout << "Activation Kernel Checks End." << std::endl;
    return 0;
}
