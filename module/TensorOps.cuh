#ifndef TENSOR_OPS_CUH
#define TENSOR_OPS_CUH
#include "Tensor.cuh"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_matmul_backward_a(Tensor *a, Tensor *b, Tensor *grad_out);
Tensor *tensor_matmul_backward_b(Tensor *a, Tensor *b, Tensor *grad_out);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *x, Tensor *bias);
void fill_tensor_with_random(Tensor *t);
Tensor *tensor_clone(Tensor *t);
void tensor_print_graph_dot(Tensor *self);

#endif // TENSOR_OPS_CUHmake