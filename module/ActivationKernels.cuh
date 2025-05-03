#ifndef ACTIVATION_KERNELS_H
#define ACTIVATION_KERNELS_H

__global__ void sigmoidKernel(float *input, float *output, int size);

__global__ void swishKernel(float *input, float *output, int size);

__global__ void reluKernel(float *input, float *output, int size);

__global__ void leakyReluKernel(float *input, float *output, float alpha, int size);

__global__ void eluKernel(float *input, float *output, float alpha, int size);

__global__ void softmaxKernel(float *input, float *output, int size);

__global__ void tanhKernel(float *input, float *output, int size);

__global__ void geluKernel(float *input, float *output, int size);

__global__ void sigmoidBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

__global__ void swishBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

__global__ void reluBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

__global__ void leakyReluBackwardKernel(float *input, float *grad_output, float *grad_input, float alpha, int size);

__global__ void eluBackwardKernel(float *input, float *grad_output, float *grad_input, float alpha, int size);

__global__ void softmaxBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

__global__ void tanhBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

__global__ void geluBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

#endif
