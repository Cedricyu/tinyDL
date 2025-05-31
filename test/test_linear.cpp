#include <stdio.h>
#include <cmath>
#include <ctime>
#include "linear.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "test.h"
#include "visualize.h"

#define VISUALIZE 1

void create_one_hot(float *target, int label, int num_classes) {
    for (int i = 0; i < num_classes; i++) {
        target[i] = (i == label) ? 1.0f : 0.0f;
    }
}

void softmax(float *logits, float *output, int size) {
    float max_logit = logits[0];
    for (int i = 1; i < size; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(logits[i] - max_logit);
        sum_exp += output[i];
    }

    for (int i = 0; i < size; i++) output[i] /= sum_exp;
}

void compute_grad_logits(float *logits, float *targets, float *grad_logits, int len) {
    float softmax_out[10];
    softmax(logits, softmax_out, len);

    for (int i = 0; i < len; i++) {
        grad_logits[i] = softmax_out[i] - targets[i];
    }
}

int argmax(float *arr, int size) {
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

void generate_data_classification(float *inputs, int *targets, int batch_size) {
    std::srand(std::time(nullptr));
    const int num_squares = 2;  // 棋盤格的大小，調整這裡可以改變複雜度
    for (int i = 0; i < batch_size; ++i) {
        float x1 = 2.0f * float(std::rand()) / RAND_MAX - 1.0f;  // [-1,1]
        float x2 = 2.0f * float(std::rand()) / RAND_MAX - 1.0f;  // [-1,1]
        inputs[i * 2 + 0] = x1;
        inputs[i * 2 + 1] = x2;

        // 將 x1, x2 映射到 0~num_squares 的格子
        int grid_x = int((x1 + 1.0f) * (num_squares / 2));
        int grid_y = int((x2 + 1.0f) * (num_squares / 2));

        // 奇偶性決定標籤：棋盤格圖案
        targets[i] = (grid_x + grid_y) % 2;
    }
}

class Model {
  public:
    Linear *linear1;
    Linear *linear2;
    Linear *linear3;
    Linear *linear4;

    Model(int input_dim, int h1, int h2, int h3, int output_dim);
    ~Model();

    Tensor *forward(Tensor *x);
    void backward(Tensor *output, float *grad_output, int batch_size, int output_dim);
    void update(float lr);
};

Model::Model(int input_dim, int h1, int h2, int h3,int output_dim) {
    linear1 = new Linear(input_dim, h1);
    linear2 = new Linear(h1, h2);
    linear3 = new Linear(h2, h3);
    linear4 = new Linear(h3, output_dim);
}

Model::~Model() {
    delete linear1;
    delete linear2;
    delete linear3;
    delete linear4;
}

Tensor *Model::forward(Tensor *x) {
    Tensor *h1 = linear1->forward(x);
    Tensor *a1 = tensor_relu(h1);
    Tensor *h2 = linear2->forward(a1);
    Tensor *a2 = tensor_relu(h2);
    Tensor *h3 = linear3->forward(a2);
    Tensor *a3 = tensor_relu(h3);
    Tensor *y = linear4->forward(a3);
    return y;
}

void Model::backward(Tensor *output, float *grad_output, int batch_size, int output_dim) {
    Tensor *grad = tensor_from_data(grad_output, batch_size, output_dim);
    tensor_backward(output, grad);
}

void Model::update(float lr) {
    tensor_update(linear4->_tensor(), lr);
    tensor_update(linear3->_tensor(), lr);
    tensor_update(linear2->_tensor(), lr);
    tensor_update(linear1->_tensor(), lr);
}

void test_linear() {
    const int input_dim = 2;
    const int hidden_dim1 = 8;
    const int hidden_dim2 = 16;
    const int hidden_dim3 = 8;
    const int output_dim = 2;
    const int total_data = 1024;
    const int epochs = 30;

    float all_inputs[total_data * input_dim] = {0.0f};
    int all_targets[total_data] = {0};

    generate_data_classification(all_inputs, all_targets, total_data);

    const int batch_size = 32;

    Model linear_model(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim);

    // // 在 Model linear_model 宣告後立即印初始權重
    printf("=== Initial Weights and Biases ===\n");
    tensor_print(linear_model.linear1->_tensor());  // 印 linear1 權重
    tensor_print(linear_model.linear2->_tensor());  // 印 linear2 權重
    tensor_print(linear_model.linear3->_tensor());  // 印 linear3 權重
    tensor_print(linear_model.linear4->_tensor());  // 印 linear4 權重
    printf("=== ============================== ===\n");

    int all_preds[total_data] = {0};  // 🔥 收集最終預測結果

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        for (int i = 0; i < total_data; i += batch_size) {
            float inputs[batch_size * input_dim] = {0.0f};
            int targets[batch_size] = {0};
            for (int j = 0; j < batch_size; ++j) {
                int idx = i + j;
                inputs[j * 2 + 0] = all_inputs[idx * 2 + 0];
                inputs[j * 2 + 1] = all_inputs[idx * 2 + 1];
                targets[j] = all_targets[idx];
            }

            Tensor *x = tensor_from_data(inputs, batch_size, input_dim);
            x->requires_grad = 1;

            Tensor *y = linear_model.forward(x);
            // tensor_print(y);

            float grad_output[batch_size * output_dim];

            float batch_loss = 0.0f;  // 🔥 初始化 batch_loss
            int correct = 0;          // 🔥 初始化 batch correct

            for (int j = 0; j < batch_size; ++j) {
                float *logits = y->data + j * output_dim;
                float probs[output_dim];
                softmax(logits, probs, output_dim);

                int label = targets[j];
                batch_loss += -logf(probs[label] + 1e-7f);

                for (int k = 0; k < output_dim; ++k) {
                    grad_output[j * output_dim + k] = probs[k] - ((k == label) ? 1.0f : 0.0f);
                }

                int pred = argmax(probs, output_dim);
                if (pred == label) correct++;
            }

            Tensor *grad = tensor_from_data(grad_output, batch_size, output_dim);

            linear_model.backward(y, grad_output, batch_size, output_dim);
            linear_model.update(0.001f);  // 更新權重

            // 累加 epoch 的 loss 和正確數
            epoch_loss += batch_loss;
            epoch_correct += correct;

            // 印出 batch 訊息
            printf("Epoch %d, Batch %d: Loss=%.4f, Accuracy=%.4f\n", epoch, i / batch_size, batch_loss / batch_size,
                   (float)correct / batch_size);

            // tensor_free(y);
        }

        // 印出 epoch 訊息
        printf("=== Epoch %d Summary: Avg Loss=%.4f, Accuracy=%.4f ===\n\n", epoch, epoch_loss / total_data,
               (float)epoch_correct / total_data);
    }

#if VISUALIZE
    for (int i = 0; i < total_data; i += batch_size) {
        int curr_batch = ((i + batch_size) <= total_data) ? batch_size : (total_data - i);
        float inputs[curr_batch * input_dim];
        for (int j = 0; j < curr_batch; ++j) {
            int idx = i + j;
            inputs[j * input_dim + 0] = all_inputs[idx * input_dim + 0];
            inputs[j * input_dim + 1] = all_inputs[idx * input_dim + 1];
        }

        Tensor *input_tensor = tensor_from_data(inputs, curr_batch, input_dim);
        Tensor *output_tensor = linear_model.forward(input_tensor);

        // tensor_print(output_tensor);  // Debug

        for (int j = 0; j < curr_batch; ++j) {
            float *logits = output_tensor->data + j * output_dim;
            float probs[output_dim];
            softmax(logits, probs, output_dim);
            int pred = argmax(probs, output_dim);
            all_preds[i + j] = pred;
        }
    }

    // for (int i = 0; i < total_data; ++i) {
    //     printf("Input: (%.2f, %.2f), Target: %d, Predicted: %d\n", all_inputs[i * 2 + 0], all_inputs[i * 2 + 1],
    //            all_targets[i], all_preds[i]);
    // }

    visualize_sdl(all_inputs, all_targets, total_data);
    visualize_sdl(all_inputs, all_preds, total_data);
#endif
}
