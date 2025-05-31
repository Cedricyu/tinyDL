#ifndef CONV2DK_CUH
#define CONV2DK_CUH

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernels
__global__ void conv2d_forward_kernel(const float *x,
                                      const float *w,
                                      float *y,
                                      int batch_size,
                                      int height,
                                      int width,
                                      int kernel_h,
                                      int kernel_w,
                                      int out_h,
                                      int out_w,
                                      int padding_h,
                                      int padding_w,
                                      int stride_h,
                                      int stride_w);

__global__ void conv2d_backward_input_kernel(const float *grad_out,
                                             const float *w,
                                             float *grad_x,
                                             int batch_size,
                                             int height,
                                             int width,
                                             int kernel_h,
                                             int kernel_w,
                                             int out_h,
                                             int out_w,
                                             int padding_h,
                                             int padding_w,
                                             int stride_h,
                                             int stride_w);

__global__ void conv2d_backward_weight_kernel(const float *grad_out,
                                              const float *x,
                                              float *grad_w,
                                              int batch_size,
                                              int height,
                                              int width,
                                              int kernel_h,
                                              int kernel_w,
                                              int out_h,
                                              int out_w,
                                              int padding_h,
                                              int padding_w,
                                              int stride_h,
                                              int stride_w);

#ifdef __cplusplus
}
#endif

#endif // CONV2DK_CUH
