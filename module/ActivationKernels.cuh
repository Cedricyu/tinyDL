#ifndef ACTIVATION_KERNELS_H
#define ACTIVATION_KERNELS_H

__global__ void sigmoidKernel(float *input, float *output, int size);

__global__ void reluKernel(float *input, float *output, int size);

__global__ void softmaxKernel(float *input, float *output, int size);

__global__ void sigmoidBackwardKernel(float *output, float *grad_output, float *grad_input, int size);

__global__ void reluBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

__global__ void softmaxBackwardKernel(float *input, float *grad_output, float *grad_input, int size);

#endif
