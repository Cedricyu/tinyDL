#ifndef TENSOR_OPS_CUH
#define TENSOR_OPS_CUH
#include "Tensor.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h> 

void tensor_matmul_backward_a(Tensor *self, Tensor *grad_out);
void tensor_matmul_backward_b(Tensor *self, Tensor *grad_out);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *x, Tensor *bias);
void fill_tensor_with_random(Tensor *t);
#endif // TENSOR_OPS_CUHmake 