#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "conv2d.cuh"
#include "linear.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"

#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE (32 * 32 * 3)
#define RECORD_SIZE (1 + IMAGE_SIZE)

// float *parse_cifar10(const char *filename, int pictureno, int *label_out) {
//     FILE *file = fopen(filename, "rb");
//     if (!file) {
//         printf("Cannot open %s\n", filename);
//         return NULL;
//     }

//     long offset = (long)pictureno * RECORD_SIZE;
//     if (fseek(file, offset, SEEK_SET) != 0) {
//         printf("Failed to seek to picture %d\n", pictureno);
//         fclose(file);
//         return NULL;
//     }

//     unsigned char buffer[RECORD_SIZE];
//     if (fread(buffer, 1, RECORD_SIZE, file) != RECORD_SIZE) {
//         printf("Failed to read picture %d\n", pictureno);
//         fclose(file);
//         return NULL;
//     }

//     unsigned char label = buffer[0];
//     if (label_out) *label_out = label;
//     // printf("Picture %d: Label=%d\n", pictureno, label);
//     unsigned char *r = buffer + 1;
//     unsigned char *g = buffer + 1 + 32*32;
//     unsigned char *b = buffer + 1 + 32*32*2;

//     static float img_data[IMAGE_SIZE];
//     for (int i = 0; i < 32 * 32; i++) {
//         img_data[i*3 + 0] = r[i] / 255.0f;  // R
//         img_data[i*3 + 1] = g[i] / 255.0f;  // G
//         img_data[i*3 + 2] = b[i] / 255.0f;  // B
//     }

//     fclose(file);
//     return img_data;
// }

// int argmax(float *arr, int size) {
//     int max_idx = 0;
//     for (int i = 1; i < size; i++) {
//         if (arr[i] > arr[max_idx]) {
//             max_idx = i;
//         }
//     }
//     return max_idx;
// }

// // Tensor *cross_entropy_loss(Tensor *logits, Tensor *targets, int num_classes) {
// //     Tensor *loss = tensor_create(1, 1, 1);  // 標量 Tensor
// //     loss->data[0] = 0.0f;

// //     // 計算 softmax
// //     float softmax_out[10];
// //     for (int i = 0; i < num_classes; i++)
// //         softmax_out[i] = logits->data[i];
// //     softmax(softmax_out, num_classes);

// //     for (int i = 0; i < num_classes; i++) {
// //         if (targets->data[i] > 0)
// //             loss->data[0] -= targets->data[i] * logf(softmax_out[i] + 1e-9f);
// //     }

// //     return loss;
// // }

// void create_one_hot(float *target, int label, int num_classes) {
//     for (int i = 0; i < num_classes; i++) {
//         target[i] = (i == label) ? 1.0f : 0.0f;
//     }
// }

// void softmax(float *logits, float *output, int size) {
//     float max_logit = logits[0];
//     for (int i = 1; i < size; i++)
//         if (logits[i] > max_logit) max_logit = logits[i];

//     float sum_exp = 0.0f;
//     for (int i = 0; i < size; i++) {
//         output[i] = expf(logits[i] - max_logit);
//         sum_exp += output[i];
//     }

//     for (int i = 0; i < size; i++)
//         output[i] /= sum_exp;
// }

// void compute_grad_logits(float *logits, float *targets, float *grad_logits, int len) {
//     float softmax_out[10];
//     softmax(logits, softmax_out, len);

//     for (int i = 0; i < len; i++) {
//         grad_logits[i] = softmax_out[i] - targets[i];
//     }
// }

// void test_conv2d() {
//     int label;

//     float w_data[] = {
//         1, 0, -1,
//         1, 0, -1,
//         1, 0, -1
//     };
//     Tensor *w = tensor_from_data(w_data, 1, 3*3);
//     w->requires_grad = 1;  // Enable gradient tracking for the weight tensor
//     Conv2D conv2d = Conv2D(w, 3, 3, 1, 1, 1);
//     int in_features = 3*3 ,hidden_features = 16, out_features = 10;
//     Linear linear1 = Linear(in_features, hidden_features);
//     Linear linear2 = Linear(hidden_features, out_features);
//     for (int pictureno = 0; pictureno < 5; pictureno++) {
//         float *image = parse_cifar10("./data/cifar-10-batches-bin/data_batch_1.bin", pictureno, &label);
//         if (!image) break;

//         Tensor *x = tensor_from_data(image, 1, 32*32*3);
//         x->requires_grad = 1;
//         Tensor *conv = conv2d.forward(x);
//         Tensor *h = linear1.forward(conv);
//         Tensor *a1 = tensor_relu(h);
//         Tensor *logits = linear2.forward(a1);

//         float targets[10];
//         create_one_hot(targets, label, 10);
//         float grad_logits[10];
//         compute_grad_logits(logits->data, targets, grad_logits, 10);
//         printf("Label: %d, Predicted: %d\n", label, argmax(logits->data, 10));

//         Tensor *grad_logits_tensor = tensor_from_data(grad_logits, 1, 10);
//         tensor_backward(logits, grad_logits_tensor);
//         // linear2.print_grad();
//         // printf("==========================\n");
//         // tensor_print_grad(a1);
//         // printf("==========================\n");
//         // linear1.print_grad();
//         // conv2d.print_grad();
//         tensor_update(linear2._tensor(), 0.001f);
//         tensor_update(linear1._tensor(), 0.001f);
//     }
// }
