#include "convkernels.cuh"

__global__ void conv2d_forward_kernel(const float *x,  // (B, C_in, H, W)
                                      const float *w,  // (C_out, C_in, K, K)
                                      float *y,        // (B, C_out, H_out, W_out)
                                      int B, int C_in, int H, int W, int K, int C_out, int H_out, int W_out,
                                      int padding, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;  // Flattened (b, c_out)
    if (out_x >= W_out || out_y >= H_out) return;

    int b = bc / C_out;
    int c_out = bc % C_out;

    float acc = 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int in_y = out_y * stride - padding + i;
                int in_x = out_x * stride - padding + j;

                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int x_idx = ((b * C_in + c_in) * H + in_y) * W + in_x;
                    int w_idx = ((c_out * C_in + c_in) * K + i) * K + j;
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }

    int y_idx = ((b * C_out + c_out) * H_out + out_y) * W_out + out_x;
    y[y_idx] = acc;
}
