#ifndef CONV2DK_CUH
#define CONV2DK_CUH

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void conv2d_forward_kernel(const float *x,  // (B, C_in, H, W)
                                      const float *w,  // (C_out, C_in, K, K)
                                      float *y,        // (B, C_out, H_out, W_out)
                                      int B, int C_in, int H, int W, int K, int C_out, int H_out, int W_out,
                                      int padding, int stride);

#ifdef __cplusplus
}
#endif

#endif  // CONV2DK_CUH
