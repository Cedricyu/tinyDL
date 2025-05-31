#include <cuda_runtime.h>
#include <stdio.h>
#include "tensor.cuh"

Tensor *tensor_create(int batch, int feat, int requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->batch_size = batch;
    t->features = feat;
    t->requires_grad = requires_grad;
    t->data = (float *)calloc(batch * feat, sizeof(float));
    t->grad = requires_grad ? (float *)calloc(batch * feat, sizeof(float)) : NULL;
    t->deps = NULL;
    t->num_deps = 0;
    return t;
}

Tensor *tensor_from_data(float *external_data, int batch, int feat) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->data = external_data;
    t->grad = NULL;
    t->batch_size = batch;
    t->features = feat;
    t->requires_grad = 0;
    t->deps = NULL;
    t->num_deps = 0;
    return t;
}

Tensor *tensor_flatten(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return NULL;
    }
    Tensor *out = tensor_create(t->batch_size, t->batch_size * t->features, t->requires_grad);
    for (int i = 0; i < t->batch_size; ++i) {
        for (int j = 0; j < t->features; ++j) {
            out->data[i * out->features + j] = t->data[i * t->features + j];
        }
    }
    return out;
}

void tensor_zero_grad(Tensor *t) {
    if (t->requires_grad && t->grad) memset(t->grad, 0, sizeof(float) * t->batch_size * t->features);
}

void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn) {
    t->deps = (Dependency *)realloc(t->deps, sizeof(Dependency) * (t->num_deps + 1));
    t->deps[t->num_deps].tensor = dep_tensor;
    t->deps[t->num_deps].backward_fn = fn;
    t->num_deps++;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->grad) free(t->grad);
    if (t->data) free(t->data);
}

void tensor_print_dimesions(Tensor *self) {
    if (!self) {
        printf("Tensor is NULL\n");
        return;
    }
    printf("Tensor dimensions: batch_size=%d, features=%d\n", self->batch_size, self->features);
}

void tensor_update(Tensor *t, float lr) {
    if (!t->requires_grad || !t->grad) {
        printf("Tensor has no gradient to update.\n");
        return;
    }

    int size = t->batch_size * t->features;
    for (int i = 0; i < size; i++) {
        t->data[i] += lr * t->grad[i];  // 梯度下降
    }
}

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return;
    }
    printf("Tensor data: ");
    printf("[");
    for(int i = 0; i < t->batch_size; ++i) {
        printf("[");
        for (int j = 0; j < t->features; ++j) {
            printf("%f ", t->data[i * t->features + j]);
        }
        printf("]");
    }
    printf("]");
    printf("\n");
}