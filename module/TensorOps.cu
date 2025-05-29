#include "CudaKernels.cuh"
#include "TensorOps.cuh"

#include <cuda_runtime.h>

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

void tensor_zero_grad(Tensor *t) {
    if (t->requires_grad && t->grad)
        memset(t->grad, 0, sizeof(float) * t->batch_size * t->features);
}

void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn) {
    t->deps = (Dependency *)realloc(t->deps, sizeof(Dependency) * (t->num_deps + 1));
    t->deps[t->num_deps].tensor = dep_tensor;
    t->deps[t->num_deps].backward_fn = fn;
    t->num_deps++;
}

void tensor_free(Tensor *t) {
    if (!t)
        return;
    if (t->grad)
        free(t->grad);
    if (t->data)
        free(t->data);
}

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
    cudaDeviceSynchronize(); // 🔥 必須加

    // Copy result back
    cudaMemcpy(out->data, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        printf("dout[%d] = %f\n", i, out->data[i]);
    }

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
    if (!self->requires_grad)
        return;

    if (self->num_deps != 2) {
        printf("Not enough dependencies found for tensor backward.\n");
        return;
    }
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
        t->grad[i] += grad->data[i]; // 假設 grad 是一個 Tensor，包含梯度數據
    }
}

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return;
    }
    printf("Tensor data: ");
    for (int i = 0; i < t->batch_size * t->features; ++i) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
}

Tensor tensor_add_bias_backward(Tensor *a, Tensor *bias, Tensor *grad_out) {
    int batch = grad_out->batch_size;
    int feat = grad_out->features;

    if (!bias->grad)
        bias->grad = (float *)calloc(1 * feat, sizeof(float));

    float *d_grad_out, *d_grad_bias;
    cudaMalloc(&d_grad_out, batch * feat * sizeof(float));
    cudaMalloc(&d_grad_bias, feat * sizeof(float));

    cudaMemcpy(d_grad_out, grad_out->grad, batch * feat * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (feat + blockSize - 1) / blockSize;

    biasGradientKernel<<<gridSize, blockSize>>>(d_grad_out, d_grad_bias, batch, feat);

    cudaMemcpy(bias->grad, d_grad_bias, feat * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_grad_out);
    cudaFree(d_grad_bias);
}

Tensor *tensor_add_bias(Tensor *x, Tensor *bias) {
    if (bias->batch_size != 1 || bias->features != x->features) {
        printf("Bias shape must be (1, features)\n");
        return NULL;
    }

    Tensor *out = tensor_create(x->batch_size, x->features, x->requires_grad || bias->requires_grad);
    memcpy(out->data, x->data, x->batch_size * x->features * sizeof(float));

    // Launch addBiasKernel
    float *d_out, *d_bias;
    cudaMalloc(&d_out, out->batch_size * out->features * sizeof(float));
    cudaMalloc(&d_bias, out->features * sizeof(float));

    cudaMemcpy(d_out, out->data, out->batch_size * out->features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias->data, out->features * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((out->features + 15) / 16, (out->batch_size + 15) / 16);
    addBiasKernel<<<gridDim, blockDim>>>(d_out, d_bias, out->batch_size, out->features);

    cudaMemcpy(out->data, d_out, out->batch_size * out->features * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_bias);

    // Register dependencies
    // if (bias->requires_grad)
    //     tensor_add_dependency(out, bias, tensor_add_bias_backward);

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
        t->data[i] = (rand_int / (double)UINT32_MAX);
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