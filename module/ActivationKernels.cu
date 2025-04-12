#include "ActivationKernels.cuh"

__global__ void sigmoidKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void swishKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
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

__global__ void eluKernel(float *input, float *output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0) ? x : alpha * (expf(x) - 1);
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

__global__ void geluKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Constants for GELU approximation
        const float sqrt_2_over_pi = 0.7978845608f;
        const float c = 0.044715f;
        float z = sqrt_2_over_pi * (x + c * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(z));
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

__global__ void swishBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float derivative = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
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

__global__ void eluBackwardKernel(float *input, float *grad_output, float *grad_input, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float derivative = (x > 0) ? 1 : alpha * expf(x);
        grad_input[idx] = grad_output[idx] * derivative;
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

__global__ void geluBackwardKernel(float *input, float *grad_output, float *grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Constants for GELU approximation
        const float sqrt_2_over_pi = 0.7978845608f;
        const float c = 0.044715f;
        float z = sqrt_2_over_pi * (x + c * x * x * x);
        float tanh_z = tanhf(z);
        float sech_z_sq = 1.0f - tanh_z * tanh_z;
        float dz_dx = sqrt_2_over_pi * (1.0f + c * 3.0f * x * x);
        float derivative = 0.5f * (1.0f + tanh_z) + 0.5f * x * sech_z_sq * dz_dx;
        grad_input[idx] = grad_output[idx] * derivative;
    }
}