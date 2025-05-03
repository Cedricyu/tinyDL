// module/Tensor.h (C-style, no STL)
#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "float_precision.h"

typedef struct Tensor Tensor;
typedef struct Dependency Dependency;

typedef void (*BackwardFn)(Tensor *target_tensor, Tensor *grad_output);

struct Dependency {
    Tensor *tensor;
    Tensor *from_output;          // out
    BackwardFn backward_fn;
};

struct Tensor {
    float_t *data;
    float_t *grad;
    int batch_size;
    int features;
    int requires_grad;

    Dependency *deps;
    int num_deps;
};

Tensor *tensor_create(int batch, int feat, int requires_grad);
Tensor *tensor_from_data(float *external_data, int batch, int feat);
void tensor_zero_grad(Tensor *t);
void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn);
void tensor_backward(Tensor *t);
void tensor_free(Tensor *t);
void tensor_print(Tensor *t);
void tensor_print_grad(Tensor *t);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
