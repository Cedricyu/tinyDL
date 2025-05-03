#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "Tensor.cuh"       // 你自己的 Tensor 結構定義
#include "TensorOps.cuh"    // 包含 tensor_matmul, tensor_backward 等

// 產生 [rows x cols] 隨機矩陣
void fill_random(Tensor *t) {
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

void test_tensor() {
    srand((unsigned int)time(NULL));

    int M = 2, K = 3, N = 4;  // x: M×K, y: K×N

    Tensor *x = tensor_create(M, K, 1);  // requires_grad = 1
    Tensor *y = tensor_create(K, N, 1);  // requires_grad = 1

    fill_random(x);
    fill_random(y);

    tensor_print(x);
    tensor_print(y);

    Tensor *z = tensor_matmul(x, y);  // z = x × y

    memcpy(z->grad, z->data, sizeof(float) * z->batch_size * z->features);
    tensor_print(z) ;
    tensor_print_grad(z);
    tensor_backward(z);  // ⬅ 會自動計算 x.grad 與 y.grad

    printf("\nx.grad:\n");
    for (int i = 0; i < M * K; ++i) printf("%.1f ", x->grad[i]);
    printf("\n");

    printf("y.grad:\n");
    for (int i = 0; i < K * N; ++i) printf("%.1f ", y->grad[i]);
    printf("\n");

    tensor_free(x);
    tensor_free(y);
    tensor_free(z);

    return ;
}
