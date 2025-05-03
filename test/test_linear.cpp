#include "Linear.cuh"
#include "Tensor.cuh"
#include "TensorOps.cuh"
#include <stdio.h>
#include "test.h"

void test_linear() {
    const int batch_size = 2;
    const int in_features = 4;
    const int hidden_features = 5;
    const int out_features = 3;

    // 建立輸入 Tensor 並初始化資料
    Tensor *x = tensor_create(batch_size, in_features, 1);  // requires_grad = 1
    fill_tensor_with_random(x);

    tensor_print(x);

    Linear linear1 = Linear(in_features, hidden_features);
    Linear linear2 = Linear(hidden_features, out_features);    

    linear1.print_weight("linear1");
    linear2.print_weight("linear2");

    // 前向傳遞
    Tensor *h = linear1.forward(x);   // h = x @ W1 + b1
    Tensor *y = linear2.forward(h);   // y = h @ W2 + b2

    printf("Final Output:\n");
    for (int i = 0; i < batch_size * out_features; ++i) {
        printf("%f ", y->data[i]);
    }
    printf("\n");

    // 模擬 output 的梯度 (手動設置 dL/dy)
    y->requires_grad = 1;
    y->grad = (float *)calloc(batch_size * out_features, sizeof(float));
    memcpy(y->grad, y->data, sizeof(float) * y->batch_size * y->features);
    tensor_backward(y);  // 自動反傳至 linear2, linear1, x

    linear2.print_grad("linear2");
    linear1.print_grad("linear1");

    printf("\n");

}