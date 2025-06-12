#include <stdio.h>
#include <stdlib.h>
#include "conv2d.cuh"
#include "convkernels.cuh"

#include "conv2d.cuh"
#include "convkernels.cuh"
#include <stdio.h>
#include <stdlib.h>

Tensor *conv2d_forward(Tensor *x, Tensor *w, int padding, int stride){
    if (x->ndim != 4 || w->ndim != 4) {
        printf("conv2d_forward only supports 4D tensors.\n");
        return NULL;
    }

    int B = x->shape[0];     // batch size
    int C_in = x->shape[1];  // input channels
    int H = x->shape[2];     // height
    int W = x->shape[3];     // width

    int C_out = w->shape[0];  // output channels
    int K = w->shape[2];      // kernel size

    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    int y_shape[] = {B, C_out, H_out, W_out};
    Tensor *y = tensor_create(4, y_shape, x->requires_grad || w->requires_grad);

    size_t size_x = sizeof(float) * tensor_numel(x->ndim, x->shape);
    size_t size_w = sizeof(float) * tensor_numel(w->ndim, w->shape);
    size_t size_y = sizeof(float) * tensor_numel(y->ndim, y->shape);

    float *d_x, *d_w, *d_y;
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_w, size_w);
    cudaMalloc(&d_y, size_y);

    cudaMemcpy(d_x, x->data, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w->data, size_w, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W_out + block.x - 1) / block.x,
              (H_out + block.y - 1) / block.y,
              B * C_out);

    conv2d_forward_kernel<<<grid, block>>>(
        d_x, d_w, d_y,
        B, C_in, H, W, K,
        C_out, H_out, W_out,
        padding, stride
    );

    cudaMemcpy(y->data, d_y, size_y, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);

    return y;
}

