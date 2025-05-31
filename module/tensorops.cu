#include "activationkernels.cuh"
#include "cudakernels.cuh"
#include "tensorops.cuh"

#include <cuda_runtime.h>

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    int M = a->batch_size;
    int K = a->features;
    int N = b->features;

    if (K != b->batch_size) {
        printf("Shape mismatch in tensor_matmul: (%d x %d) × (%d x %d)\n", M, K, b->batch_size, N);
        return NULL;
    }

    Tensor *out = tensor_create(M, N, a->requires_grad || b->requires_grad);

    // Device memory
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));

    cudaMemcpy(d_a, a->data, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, M, K, N);
    cudaDeviceSynchronize();  // 🔥 必須加

    // Copy result back
    cudaMemcpy(out->data, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < M * N; i++) {
    //     printf("dout[%d] = %f\n", i, out->data[i]);
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // AutoGrad
    if (a->requires_grad || b->requires_grad) {
        tensor_add_dependency(out, a, tensor_matmul_backward_a);
        tensor_add_dependency(out, b, tensor_matmul_backward_b);
    }

    return out;
}

Tensor *tensor_matmul_backward_a(Tensor *a, Tensor *b, Tensor *grad_out) {
    int M = a->batch_size;
    int K = a->features;
    int N = b->features;

    Tensor *grad_a = tensor_create(M, K, 0);  // 新建 Tensor 儲存梯度

    // GPU 記憶體配置
    float *d_grad_out, *d_b, *d_grad_a;
    cudaMalloc(&d_grad_out, M * N * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_grad_a, M * K * sizeof(float));

    cudaMemcpy(d_grad_out, grad_out->data, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // GPU: 轉置 b 並相乘
    float *d_b_T;
    cudaMalloc(&d_b_T, N * K * sizeof(float));
    dim3 blockDim(16, 16);
    dim3 gridDim((K + 15) / 16, (N + 15) / 16);
    matrixTransposeKernel<<<gridDim, blockDim>>>(d_b, d_b_T, K, N);
    cudaDeviceSynchronize();

    gridDim = dim3((K + 15) / 16, (M + 15) / 16);
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_grad_out, d_b_T, d_grad_a, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(grad_a->data, d_grad_a, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_grad_out);
    cudaFree(d_b);
    cudaFree(d_b_T);
    cudaFree(d_grad_a);

    return grad_a;
}

Tensor *tensor_matmul_backward_b(Tensor *a, Tensor *b, Tensor *grad_out) {
    int M = a->batch_size;
    int K = a->features;
    int N = b->features;

    Tensor *grad_b = tensor_create(K, N, 0);  // 新建 Tensor 儲存梯度

    float *d_a, *d_a_T, *d_grad_out, *d_grad_b;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_a_T, K * M * sizeof(float));
    cudaMalloc(&d_grad_out, M * N * sizeof(float));
    cudaMalloc(&d_grad_b, K * N * sizeof(float));

    cudaMemcpy(d_a, a->data, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_out, grad_out->data, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((M + 15) / 16, (K + 15) / 16);
    matrixTransposeKernel<<<gridDim, blockDim>>>(d_a, d_a_T, M, K);
    cudaDeviceSynchronize();

    gridDim = dim3((N + 15) / 16, (K + 15) / 16);
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_a_T, d_grad_out, d_grad_b, K, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(grad_b->data, d_grad_b, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_a_T);
    cudaFree(d_grad_out);
    cudaFree(d_grad_b);

    return grad_b;
}

void tensor_backward(Tensor *self, Tensor *grad_out) {
    if (!self->requires_grad) return;

    if (self->num_deps == 2) {
        Dependency *dep0 = &self->deps[0];
        Dependency *dep1 = &self->deps[1];
        if (dep0->backward_fn) {
            Tensor *grad_a = dep0->backward_fn(dep0->tensor, dep1->tensor, grad_out);
            tensor_grad(dep0->tensor, grad_a);
            tensor_backward(dep0->tensor, grad_a);
            tensor_free(grad_a);
        }
        if (dep1 && dep1->backward_fn) {
            Tensor *grad_b = dep1->backward_fn(dep0->tensor, dep1->tensor, grad_out);
            tensor_grad(dep1->tensor, grad_b);
            tensor_free(grad_b);
        }
    } else if (self->num_deps == 1) {
        Dependency *dep = &self->deps[0];
        if (dep->backward_fn) {
            Tensor *grad = dep->backward_fn(dep->tensor, NULL, grad_out);
            tensor_grad(dep->tensor, grad);
            tensor_backward(dep->tensor, grad);
            tensor_free(grad);
        }
    } else {
        printf("No dependencies found for backward pass.\n");
    }
}

void tensor_print_graph_dot_rec(Tensor *self) {
    if (!self) return;

    printf("  \"%p\" [label=\"Tensor %p\"];\n", self, self);
    for (int i = 0; i < self->num_deps; ++i) {
        Dependency *dep = &self->deps[i];
        if (dep && dep->tensor) {
            printf("  \"%p\" -> \"%p\";\n", self, dep->tensor);
            tensor_print_graph_dot_rec(dep->tensor);
        }
    }
}

void tensor_print_graph_dot(Tensor *self) {
    printf("digraph G {\n");
    tensor_print_graph_dot_rec(self);
    printf("}\n");
}

void tensor_grad(Tensor *t, Tensor *grad) {
    if (!t->requires_grad) {
        printf("Tensor does not require gradient.\n");
        return;
    }
    if (!t->grad) {
        t->grad = (float *)calloc(t->batch_size * t->features, sizeof(float));
    }
    for (int i = 0; i < t->batch_size * t->features; ++i) {
        t->grad[i] = -grad->data[i];
    }
}

Tensor *tensor_relu_backward(Tensor *x, Tensor *n, Tensor *grad_out) {
    int size = x->batch_size * x->features;

    Tensor *grad_x = tensor_create(x->batch_size, x->features, 0);

    float *d_x, *d_grad_out, *d_grad_x;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_grad_out, size * sizeof(float));
    cudaMalloc(&d_grad_x, size * sizeof(float));

    cudaMemcpy(d_x, x->data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_out, grad_out->data, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    reluBackwardKernel<<<gridSize, blockSize>>>(d_x, d_grad_out, d_grad_x, size);
    cudaDeviceSynchronize();

    cudaMemcpy(grad_x->data, d_grad_x, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_grad_out);
    cudaFree(d_grad_x);

    return grad_x;
}

Tensor *tensor_relu(Tensor *x) {
    Tensor *out = tensor_create(x->batch_size, x->features, x->requires_grad);
    int size = x->batch_size * x->features;

    float *d_x, *d_out;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));

    cudaMemcpy(d_x, x->data, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    reluKernel<<<gridSize, blockSize>>>(d_x, d_out, size);
    cudaMemcpy(out->data, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);

    // 建立依賴鏈
    if (x->requires_grad) {
        tensor_add_dependency(out, x, tensor_relu_backward);
    }

    return out;
}

void tensor_print_grad(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return;
    }
    printf("Tensor grad: ");
    for (int i = 0; i < t->batch_size * t->features; ++i) {
        printf("%f ", t->grad[i]);
    }
    printf("\n");
}

void fill_tensor_with_random(Tensor *t) {
    FILE *fp = fopen("/dev/urandom", "rb");
    if (!fp) {
        perror("fopen");
        exit(1);
    }
    for (int i = 0; i < t->batch_size * t->features; i++) {
        u_int32_t rand_int;
        fread(&rand_int, sizeof(rand_int), 1, fp);
        t->data[i] = 2.0f * (rand_int / (double)UINT32_MAX) - 1.0f;
    }
    fclose(fp);
}

Tensor *tensor_clone(Tensor *t) {
    Tensor *clone = (Tensor *)malloc(sizeof(Tensor));
    clone->batch_size = t->batch_size;
    clone->features = t->features;
    clone->requires_grad = t->requires_grad;
    clone->data = (float *)malloc(t->batch_size * t->features * sizeof(float));
    clone->grad = t->requires_grad ? (float *)malloc(t->batch_size * t->features * sizeof(float)) : NULL;
    memcpy(clone->data, t->data, t->batch_size * t->features * sizeof(float));
    return clone;
}