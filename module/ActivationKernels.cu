#include "ActivationKernels.cuh"

__global__ void sigmoidKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void reluKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

__global__ void softmaxKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Subtract max for numerical stability
        // @TODO: A separate kernel for max value calculation would be more efficient
        float max_value = input[0];
        for (int i = 1; i < size; ++i) {
            if (input[i] > max_value) {
                max_value = input[i];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += expf(input[i] - max_value);
        }

        output[idx] = expf(input[idx] - max_value) / sum;
    }
}

__global__ void sigmoidBackwardKernel(float *output, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_out = output[idx];
        float derivative = sigmoid_out * (1.0f - sigmoid_out);
        grad_input[idx] = grad_output[idx] * derivative;
    }
}

__global__ void reluBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0;
    }
}

__global__ void softmaxBackwardKernel(float *output, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int j = 0; j < size; ++j) {
            sum += output[j] * grad_output[j];
        }
        grad_input[idx] = output[idx] * (grad_output[idx] - sum);
    }
}
