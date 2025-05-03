#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void conv2dKernel(
    float *input, float *weight, float *output, int B, int C_in, int H, int W, int C_out, int K, int outH, int outW) {
    int b = blockIdx.z;         // batch index
    int m = blockIdx.y;         // output channel
    int oh = blockIdx.x / outW; // output height
    int ow = blockIdx.x % outW; // output width

    float val = 0.0f;
    for (int c = 0; c < C_in; ++c) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int ih = oh + i;
                int iw = ow + j;
                val += input[((b * C_in + c) * H + ih) * W + iw] * weight[((m * C_in + c) * K + i) * K + j];
            }
        }
    }

    output[((b * C_out + m) * outH + oh) * outW + ow] = val;
}

class Convolution2D {
  public:
    int in_channels, out_channels, kernel_size;
    float *d_weight;

    Convolution2D(int in_c, int out_c, int k) : in_channels(in_c), out_channels(out_c), kernel_size(k) {
        size_t weight_size = in_c * out_c * k * k * sizeof(float);
        cudaMalloc(&d_weight, weight_size);

        // 初始化權重（設為 1.0f）
        float *h_weight = (float *)malloc(weight_size);
        for (int i = 0; i < in_c * out_c * k * k; ++i)
            h_weight[i] = 1.0f;
        cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
        free(h_weight);
    }

    void forward(float *input, float *output, int B, int H, int W) {
        int outH = H - kernel_size + 1;
        int outW = W - kernel_size + 1;

        float *d_input, *d_output;
        size_t input_size = B * in_channels * H * W * sizeof(float);
        size_t output_size = B * out_channels * outH * outW * sizeof(float);

        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);
        cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

        dim3 grid(outH * outW, out_channels, B);
        dim3 block(1);

        conv2dKernel<<<grid, block>>>(
            d_input, d_weight, d_output, B, in_channels, H, W, out_channels, kernel_size, outH, outW);

        cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
    }

    ~Convolution2D() {
        cudaFree(d_weight);
    }
};
