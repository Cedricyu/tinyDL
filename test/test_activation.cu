#include "ActivationKernels.cuh"
#include "CudaHelper.cuh"
#include <cuda_runtime.h>
#include <iostream>

void leaky_relu_forward(float *h_input, float *h_output, float alpha, int size) {
    float *d_input, *d_output;

    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Failed to allocate d_input.");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Failed to allocate d_output.");

    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy d_input to device.");

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    leakyReluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, alpha, size);
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

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    leakyReluBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_grad_output, d_grad_input, alpha, size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed.");

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}

void test_leaky_relu() {
    int size = 1024;
    float alpha = 0.01f;

    float *h_input = new float[size];
    float *h_output = new float[size];
    float *h_grad_input = new float[size];
    float *h_grad_output = new float[size];

    for (int i = 0; i < size; ++i) {
        h_input[i] = -2.0f + (4.0f * i) / (size - 1);
        h_grad_input[i] = 1.0f;
    }

    leaky_relu_forward(h_input, h_output, alpha, size);
    leaky_relu_backward(h_input, h_grad_output, h_grad_input, alpha, size);

    std::cout << "Leaky ReLU Check: " << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << "Input: " << h_input[i] << ", Foward Output: " << h_output[i]
                  << ", Grad Output: " << h_grad_output[i] << ", Grad Input: " << h_grad_input[i] << std::endl;
    }

    delete[] h_input;
    delete[] h_output;
    delete[] h_grad_input;
    delete[] h_grad_output;
    std::cout << "Leaky ReLU Check Passed!" << std::endl;
}
