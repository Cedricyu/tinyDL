#include "ActivationKernels.cuh"

__global__ void reluKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

__global__ void reluBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0;
    }
}
