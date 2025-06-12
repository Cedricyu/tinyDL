#include <cuda_runtime.h>
#include <stdio.h>
#include "tensor.cuh"

int tensor_numel(int ndim, int *shape) {
    int total = 1;
    for (int i = 0; i < ndim; i++) total *= shape[i];
    return total;
}

Tensor *tensor_create(int ndim, int *shape, int requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->requires_grad = requires_grad;

    t->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(t->shape, shape, sizeof(int) * ndim);

    int total_size = tensor_numel(ndim, shape);
    t->data = (float_t *)calloc(total_size, sizeof(float_t));
    t->grad = requires_grad ? (float_t *)calloc(total_size, sizeof(float_t)) : NULL;

    t->deps = NULL;
    t->num_deps = 0;
    return t;
}

Tensor *tensor_from_data(float *external_data, int ndim, int *shape) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(t->shape, shape, sizeof(int) * ndim);
    t->data = external_data;
    t->grad = NULL;
    t->requires_grad = 0;
    t->deps = NULL;
    t->num_deps = 0;
    return t;
}

Tensor *tensor_from_data_2d(float *data, int dim0, int dim1) {
    int shape[] = {dim0, dim1};
    return tensor_from_data(data, 2, shape);
}

void tensor_zero_grad(Tensor *t) {
    if (t->requires_grad && t->grad) {
        int total = tensor_numel(t->ndim, t->shape);
        memset(t->grad, 0, sizeof(float) * total);
    }
}

void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn) {
    t->deps = (Dependency *)realloc(t->deps, sizeof(Dependency) * (t->num_deps + 1));
    t->deps[t->num_deps].tensor = dep_tensor;
    t->deps[t->num_deps].backward_fn = fn;
    t->num_deps++;
}

void tensor_update(Tensor *t, float lr) {
    if (!t->requires_grad || !t->grad) {
        printf("Tensor has no gradient to update.\n");
        return;
    }

    int total = tensor_numel(t->ndim, t->shape);
    for (int i = 0; i < total; i++) {
        t->data[i] += lr * t->grad[i];  // 梯度下降
    }
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->grad) free(t->grad);
    if (t->data) free(t->data);
    if (t->shape) free(t->shape);
    if (t->deps) free(t->deps);
    free(t);
}

void tensor_print_dimensions(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return;
    }

    printf("Tensor shape: (");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%d", t->shape[i]);
        if (i != t->ndim - 1) printf(", ");
    }
    printf(")\n");
}

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return;
    }

    int total = tensor_numel(t->ndim, t->shape);
    printf("Tensor data: [");
    for (int i = 0; i < total; ++i) {
        printf("%f ", t->data[i]);
    }
    printf("]\n");
}