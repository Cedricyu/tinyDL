#ifndef TENSOR_OPS_CUH
#define TENSOR_OPS_CUH
#include "Tensor.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CudaKernels.cuh"

extern __global__ void matrixMultiplyKernel(const float *, const float *, float *, int, int, int);
extern __global__ void addBiasKernel(float *, const float *, int, int);
extern __global__ void biasGradientKernel(const float *, float *, int, int);

void tensor_matmul_backward_a(Tensor *self, Tensor *grad_out);
void tensor_matmul_backward_b(Tensor *self, Tensor *grad_out);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
#endif // TENSOR_OPS_CUH