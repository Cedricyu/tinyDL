#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "conv2d.cuh"
#include "linear.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "visualize.h"

#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE (32 * 32 * 3)
#define RECORD_SIZE (1 + IMAGE_SIZE)

float *parse_cifar10(const char *filename, int pictureno, int *label_out) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open %s\n", filename);
        return NULL;
    }

    long offset = (long)pictureno * RECORD_SIZE;
    if (fseek(file, offset, SEEK_SET) != 0) {
        printf("Failed to seek to picture %d\n", pictureno);
        fclose(file);
        return NULL;
    }

    unsigned char buffer[RECORD_SIZE];
    if (fread(buffer, 1, RECORD_SIZE, file) != RECORD_SIZE) {
        printf("Failed to read picture %d\n", pictureno);
        fclose(file);
        return NULL;
    }

    unsigned char label = buffer[0];
    if (label_out) *label_out = label;
    // printf("Picture %d: Label=%d\n", pictureno, label);
    unsigned char *r = buffer + 1;
    unsigned char *g = buffer + 1 + 32 * 32;
    unsigned char *b = buffer + 1 + 32 * 32 * 2;

    static float img_data[IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++) {
        img_data[i] = buffer[i + 1] / 255.0f;
    }

    fclose(file);
    return img_data;
}

void save_tensor_channel_as_ppm(Tensor *t, int batch_idx, int channel, int height, int width, const char *filename) {
    if (t->ndim != 4) {
        printf("save_tensor_channel_as_ppm only supports 4D tensors.\n");
        return;
    }

    int B = t->shape[0];
    int C = t->shape[1];
    int H = t->shape[2];
    int W = t->shape[3];

    if (batch_idx >= B || channel >= C || height != H || width != W) {
        printf("save_tensor_channel_as_ppm: shape mismatch or out of range.\n");
        return;
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Failed to open file %s\n", filename);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    int offset = ((batch_idx * C + channel) * H * W);
    float *data = t->data + offset;

    float min_val = 1e10f, max_val = -1e10f;
    for (int i = 0; i < height * width; i++) {
        float v = data[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    float range = max_val - min_val;
    if (fabs(range) < 1e-6f) range = 1.0f;  // avoid divide by 0

    for (int i = 0; i < height * width; i++) {
        float norm = (data[i] - min_val) / range;
        uint8_t pixel = (uint8_t)(norm * 255.0f);
        fputc(pixel, f);
        fputc(pixel, f);
        fputc(pixel, f);  // RGB 灰階輸出
    }

    fclose(f);
}

void save_rgb_image_as_ppm(const float *image, int height, int width, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Failed to open file %s\n", filename);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    // CIFAR-10 排列：前1024為R，接著G，再來B
    const float *r = image;
    const float *g = image + width * height;
    const float *b = image + 2 * width * height;

    for (int i = 0; i < width * height; i++) {
        fputc((uint8_t)(r[i] * 255.0f), f);
        fputc((uint8_t)(g[i] * 255.0f), f);
        fputc((uint8_t)(b[i] * 255.0f), f);
    }

    fclose(f);
}

void test_conv2d() {
    int label;

    const int C_out = 3;
    const int C_in = 3;
    const int kernel_size = 3;
    const int padding = 1;
    const int stride = 1;
    const int H = 32, W = 32;

    // 建立卷積核權重張量 (C_out, C_in, K, K)
    int shape_w[] = {C_out, C_in, kernel_size, kernel_size};
    Tensor *w = tensor_create(4, shape_w, 1);

    // 三種 kernel（每個 output channel 使用一種）
    float k1[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};   // Sobel x
    float k2[] = {1, 1, 1, 0, 0, 0, -1, -1, -1};   // Sobel y
    float k3[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};  // Sharpen

    for (int co = 0; co < C_out; co++) {
        float *kernel = (co == 0) ? k1 : (co == 1 ? k2 : k3);
        for (int ci = 0; ci < C_in; ci++) {
            for (int k = 0; k < kernel_size * kernel_size; k++) {
                int idx = ((co * C_in + ci) * kernel_size * kernel_size) + k;
                w->data[idx] = kernel[k];
            }
        }
    }

    for (int pictureno = 0; pictureno < 5; pictureno++) {
        float *image = parse_cifar10("./data/cifar-10-batches-bin/data_batch_1.bin", pictureno, &label);
        if (!image) break;

        // 儲存原始圖像
        char original_name[64];
        snprintf(original_name, sizeof(original_name), "images/original_%d.ppm", pictureno);
        save_rgb_image_as_ppm(image, W, H, original_name);
        printf("Saved original image: %s\n", original_name);

        // 包裝輸入影像成 Tensor (1, 3, 32, 32)
        int shape_x[] = {1, 3, H, W};
        Tensor *x = tensor_create(4, shape_x, 1);
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < H * W; i++) x->data[c * H * W + i] = image[c * H * W + i] / 255.0f;

        // 前向卷積
        Tensor *y = conv2d_forward(x, w ,padding, stride);  // output shape: (1, C_out, H, W)

        int C_result = y->shape[1];
        int H_result = y->shape[2];
        int W_result = y->shape[3];

        // 儲存每個 channel 圖像
        for (int ch = 0; ch < C_result; ch++) {
            char fname[64];
            snprintf(fname, sizeof(fname), "images/conv_out_%d_channel%d.ppm", pictureno, ch);
            save_tensor_channel_as_ppm(y, 0, ch, H_result, W_result, fname);
            printf("Saved conv channel %d image: %s\n", ch, fname);
        }
    }

    tensor_free(w);
}
