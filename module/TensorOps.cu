#include "TensorOps.cuh"

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
    t->deps[t->num_deps].from_output = t;
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

void tensor_matmul_backward_a(Tensor *a, Tensor *grad_out) {

    Tensor *out = grad_out;
    Tensor *b = out->deps[1].tensor;

    int M = a->batch_size;
    int K = a->features;
    int N = b->features;
    if (!a->grad)
        a->grad = (float *)calloc(M * K, sizeof(float));

    float *d_grad_out, *d_b, *d_grad_a;
    cudaMalloc(&d_grad_out, M * N * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_grad_a, M * K * sizeof(float));

    cudaMemcpy(d_grad_out, out->grad, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    matrixTransposeMulKernel<<<grid, blockSize>>>(d_grad_out, d_b, d_grad_a, M, N, K);

    float *h_grad_a = (float *)malloc(M * K * sizeof(float));
    cudaMemcpy(h_grad_a, d_grad_a, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * K; i++)
        a->grad[i] += h_grad_a[i];
    free(h_grad_a);
    cudaFree(d_grad_out);
    cudaFree(d_b);
    cudaFree(d_grad_a);
}

void tensor_matmul_backward_b(Tensor *b, Tensor *grad_out) {

    Tensor *out = grad_out;
    Tensor *a = out->deps[0].tensor;

    int M = a->batch_size;
    int K = a->features;
    int N = b->features;
    if (!b->grad)
        b->grad = (float *)calloc(K * N, sizeof(float));

    float *d_a, *d_grad_out, *d_grad_b;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_grad_out, M * N * sizeof(float));
    cudaMalloc(&d_grad_b, K * N * sizeof(float));

    cudaMemcpy(d_a, a->data, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_out, out->grad, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 grid((N + 15) / 16, (K + 15) / 16);
    matrixTransposeMulKernel<<<grid, blockSize>>>(d_a, d_grad_out, d_grad_b, K, M, N);

    float *h_grad_b = (float *)malloc(K * N * sizeof(float));
    cudaMemcpy(h_grad_b, d_grad_b, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < K * N; i++)
        b->grad[i] += h_grad_b[i];
    free(h_grad_b);
    cudaFree(d_a);
    cudaFree(d_grad_out);
    cudaFree(d_grad_b);
}

void tensor_backward(Tensor *self) {
    if (!self->requires_grad)
        return;

    // 如果 grad 尚未設置，設為全 1
    if (!self->grad) {
        self->grad = (float *)calloc(self->batch_size * self->features, sizeof(float));
    }

    int total = self->batch_size * self->features;
    for (int i = 0; i < total; ++i) {
        self->grad[i] = 1.0f;
    }

    // 對每個 dependency 呼叫它的 backward_fn
    for (int i = 0; i < self->num_deps; ++i) {
        Dependency *dep = &self->deps[i];
        if (dep->backward_fn) {
            dep->backward_fn(dep->tensor, self); // 傳 input 和 output
        }
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