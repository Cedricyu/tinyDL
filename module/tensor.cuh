// module/Tensor.h (C-style, no STL)
#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "float_precision.h"

typedef struct Tensor Tensor;
typedef struct Dependency Dependency;

typedef Tensor *(*BackwardFn)(Tensor *a, Tensor *b, Tensor *grad_out);

struct Dependency {
    Tensor *tensor;
    BackwardFn backward_fn;
};

struct Tensor {
    float_t *data;
    float_t *grad;
    int *shape;
    int ndim;
    int requires_grad;

    Dependency *deps;
    int num_deps;
};

Tensor *tensor_create(int ndim, int *shape, int requires_grad);
Tensor *tensor_from_data(float *external_data, int ndim, int *shape);
Tensor *tensor_from_data_2d(float *data, int dim0, int dim1);
Tensor *tensor_flatten(Tensor *t);
int tensor_numel(int ndim, int *shape);
void tensor_zero_grad(Tensor *t);
void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn);
void tensor_backward(Tensor *t, Tensor *grad_output);
void tensor_free(Tensor *t);
void tensor_print(Tensor *t);
void tensor_print_grad(Tensor *t);
void tensor_grad(Tensor *t, Tensor *grad);
void tensor_update(Tensor *t, float lr) ;

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
