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

__global__ void leakyReluKernel(float *input, float *output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0) ? x : alpha * x;
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

__global__ void tanhKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void sigmoidBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_out = input[idx];
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

__global__ void leakyReluBackwardKernel(float *input, float *grad_output, float *grad_input, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        grad_input[idx] = (x > 0) ? grad_output[idx] : alpha * grad_output[idx];
    }
}

__global__ void softmaxBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int j = 0; j < size; ++j) {
            sum += input[j] * grad_output[j];
        }
        grad_input[idx] = input[idx] * (grad_output[idx] - sum);
    }
}

__global__ void tanhBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tanh_val = tanhf(input[idx]);
        grad_input[idx] = grad_output[idx] * (1.0f - tanh_val * tanh_val);
    }
}
