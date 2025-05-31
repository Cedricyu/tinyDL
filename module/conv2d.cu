#include <stdio.h>
#include <stdlib.h>
#include "conv2d.cuh"
#include "convkernels.cuh"

Tensor *conv2d_forward(Tensor *x, Tensor *w) {
    // 先指定固定參數
    int height = 4;
    int width = 4;
    int kernel_h = 2;
    int kernel_w = 2;
    int padding_h = 0, padding_w = 0;
    int stride_h = 1, stride_w = 1;

    int out_h = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * padding_w - kernel_w) / stride_w + 1;

    Tensor *y = tensor_create(x->batch_size, out_h * out_w, x->requires_grad || w->requires_grad);

    // 分配 GPU 記憶體
    float *d_x, *d_w, *d_y;
    size_t size_x = sizeof(float) * x->batch_size * height * width;
    size_t size_w = sizeof(float) * kernel_h * kernel_w;
    size_t size_y = sizeof(float) * x->batch_size * out_h * out_w;

    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_w, size_w);
    cudaMalloc(&d_y, size_y);

    cudaMemcpy(d_x, x->data, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w->data, size_w, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y, x->batch_size);

    conv2d_forward_kernel<<<grid, block>>>(d_x, d_w, d_y, x->batch_size, height, width, kernel_h, kernel_w, out_h,
                                           out_w, padding_h, padding_w, stride_h, stride_w);

    cudaMemcpy(y->data, d_y, size_y, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);

    if (y->requires_grad) {
        tensor_add_dependency(y, x, conv2d_backward_input);
        tensor_add_dependency(y, w, conv2d_backward_weight);
    }

    return y;
}

Tensor *conv2d_backward_input(Tensor *x, Tensor *w, Tensor *grad_out) {
    // 假設這些 meta 數據固定或從 x 中獲取
    int batch_size = x->batch_size;
    int height = 4;  // 從 x->meta_height 或直接指定
    int width = 4;
    int kernel_h = 2;
    int kernel_w = 2;
    int padding_h = 0, padding_w = 0;
    int stride_h = 1, stride_w = 1;

    int out_h = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * padding_w - kernel_w) / stride_w + 1;

    Tensor *grad_x = tensor_create(batch_size, height * width, 0);

    float *d_grad_out, *d_w, *d_grad_x;
    size_t size_go = sizeof(float) * batch_size * out_h * out_w;
    size_t size_w = sizeof(float) * kernel_h * kernel_w;
    size_t size_gx = sizeof(float) * batch_size * height * width;

    cudaMalloc(&d_grad_out, size_go);
    cudaMalloc(&d_w, size_w);
    cudaMalloc(&d_grad_x, size_gx);
    cudaMemcpy(d_grad_out, grad_out->data, size_go, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w->data, size_w, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
    conv2d_backward_input_kernel<<<grid, block>>>(d_grad_out, d_w, d_grad_x, batch_size, height, width, kernel_h,
                                                  kernel_w, out_h, out_w, padding_h, padding_w, stride_h, stride_w);

    cudaMemcpy(grad_x->data, d_grad_x, size_gx, cudaMemcpyDeviceToHost);
    cudaFree(d_grad_out);
    cudaFree(d_w);
    cudaFree(d_grad_x);

    return grad_x;
}

Tensor *conv2d_backward_weight(Tensor *x, Tensor *w, Tensor *grad_out) {
    int batch_size = x->batch_size;
    int height = 4;
    int width = 4;
    int kernel_h = 2;
    int kernel_w = 2;
    int padding_h = 0, padding_w = 0;
    int stride_h = 1, stride_w = 1;

    int out_h = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * padding_w - kernel_w) / stride_w + 1;

    Tensor *grad_w = tensor_create(1, kernel_h * kernel_w, 0);

    float *d_grad_out, *d_x, *d_grad_w;
    size_t size_go = sizeof(float) * batch_size * out_h * out_w;
    size_t size_x = sizeof(float) * batch_size * height * width;
    size_t size_gw = sizeof(float) * kernel_h * kernel_w;

    cudaMalloc(&d_grad_out, size_go);
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_grad_w, size_gw);
    cudaMemcpy(d_grad_out, grad_out->data, size_go, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x->data, size_x, cudaMemcpyHostToDevice);
    cudaMemset(d_grad_w, 0, size_gw);

    dim3 block(16, 16);
    dim3 grid(kernel_w, kernel_h);
    conv2d_backward_weight_kernel<<<grid, block>>>(d_grad_out, d_x, d_grad_w, batch_size, height, width, kernel_h,
                                                   kernel_w, out_h, out_w, padding_h, padding_w, stride_h, stride_w);

    cudaMemcpy(grad_w->data, d_grad_w, size_gw, cudaMemcpyDeviceToHost);
    cudaFree(d_grad_out);
    cudaFree(d_x);
    cudaFree(d_grad_w);

    return grad_w;
}
