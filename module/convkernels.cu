#include "convkernels.cuh"

__global__ void conv2d_forward_kernel(const float *x, const float *w, float *y, int batch_size, int height, int width,
                                      int kernel_h, int kernel_w, int out_h, int out_w, int padding_h, int padding_w,
                                      int stride_h, int stride_w) {
    int b = blockIdx.z;                             // batch index
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output col

    if (i < out_h && j < out_w) {
        float sum = 0.0f;
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_h = i * stride_h + kh - padding_h;
                int in_w = j * stride_w + kw - padding_w;
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    sum += x[b * height * width + in_h * width + in_w] * w[kh * kernel_w + kw];
                }
            }
        }
        y[b * out_h * out_w + i * out_w + j] = sum;
    }
}

__global__ void conv2d_backward_input_kernel(const float *grad_out, const float *w, float *grad_x, int batch_size,
                                             int height, int width, int kernel_h, int kernel_w, int out_h, int out_w,
                                             int padding_h, int padding_w, int stride_h, int stride_w) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float sum = 0.0f;
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int out_i = (i + padding_h - kh);
                int out_j = (j + padding_w - kw);
                if (out_i % stride_h == 0 && out_j % stride_w == 0) {
                    out_i /= stride_h;
                    out_j /= stride_w;
                    if (out_i >= 0 && out_i < out_h && out_j >= 0 && out_j < out_w) {
                        sum += w[kh * kernel_w + kw] * grad_out[b * out_h * out_w + out_i * out_w + out_j];
                    }
                }
            }
        }
        grad_x[b * height * width + i * width + j] = sum;
    }
}

__global__ void conv2d_backward_weight_kernel(const float *grad_out, const float *x, float *grad_w, int batch_size,
                                              int height, int width, int kernel_h, int kernel_w, int out_h, int out_w,
                                              int padding_h, int padding_w, int stride_h, int stride_w) {
    int kh = blockIdx.y;
    int kw = blockIdx.x;

    if (kh < kernel_h && kw < kernel_w) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    int in_h = i * stride_h + kh - padding_h;
                    int in_w = j * stride_w + kw - padding_w;
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        sum +=
                            x[b * height * width + in_h * width + in_w] * grad_out[b * out_h * out_w + i * out_w + j];
                    }
                }
            }
        }
        grad_w[kh * kernel_w + kw] = sum;
    }
}