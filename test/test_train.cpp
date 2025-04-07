#include "CudaDeviceInfo.h"
#include "Linear.cuh"
#include "SGD.cuh"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
inline float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

void generate_data_classification(float *inputs, int *targets, int batch_size) {
    std::srand(std::time(nullptr));
    for (int i = 0; i < batch_size; ++i) {
        float x1 = float(std::rand()) / RAND_MAX;
        float x2 = float(std::rand()) / RAND_MAX;
        inputs[i * 2 + 0] = x1;
        inputs[i * 2 + 1] = x2;
        targets[i] = (x1 > x2) ? 1 : 0;
    }
}

void train(int epochs = 1000, int batch_size = 64) {
    const int input_dim = 2;
    const int hidden_dim = 8;
    const int output_dim = 1;
    const int total_data = 1024;

    // 資料準備
    std::vector<float> all_inputs(total_data * input_dim);
    std::vector<int> all_targets(total_data);
    generate_data_classification(all_inputs.data(), all_targets.data(), total_data);

    // 模型初始化
    Linear layer1(input_dim, hidden_dim);
    Linear layer2(hidden_dim, output_dim);
    SGD optimizer(0.0001f);
    optimizer.add_param(layer1.d_weight, layer1.d_grad_weight, input_dim * hidden_dim);
    optimizer.add_param(layer1.d_bias, layer1.d_grad_bias, hidden_dim);
    optimizer.add_param(layer2.d_weight, layer2.d_grad_weight, hidden_dim * output_dim);
    optimizer.add_param(layer2.d_bias, layer2.d_grad_bias, output_dim);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        for (int i = 0; i < total_data; i += batch_size) {
            // 取 batch 資料
            auto x = std::make_shared<Tensor>(batch_size, input_dim);
            auto y = std::vector<int>(batch_size);

            for (int j = 0; j < batch_size; ++j) {
                int idx = i + j;
                x->data[j * 2 + 0] = all_inputs[idx * 2 + 0];
                x->data[j * 2 + 1] = all_inputs[idx * 2 + 1];
                y[j] = all_targets[idx];
            }

            // Forward
            auto h = layer1.forward(x); // Tensor
            for (int j = 0; j < h->size(); ++j)
                h->data[j] = sigmoid(h->data[j]);

            auto out = layer2.forward(h);

            // Loss & grad_output
            auto grad_out = std::make_shared<Tensor>(batch_size, output_dim);
            float loss = 0.0f;
            int correct = 0;
            float eps = 1e-7f;

            for (int j = 0; j < batch_size; ++j) {
                float raw = out->data[j];
                float prob = sigmoid(raw);
                float label = static_cast<float>(y[j]);

                loss += -(label * std::log(prob + eps) + (1 - label) * std::log(1 - prob + eps));
                grad_out->data[j] = prob - label;

                if ((prob > 0.5f) == (y[j] == 1))
                    correct++;
            }

            // Backward
            auto grad_hidden = layer2.backward(h, grad_out);
            for (int j = 0; j < grad_hidden->size(); ++j)
                grad_hidden->data[j] *= sigmoid_derivative(h->data[j]);

            layer1.backward(x, grad_hidden);

            // 更新權重
            optimizer.step();
            optimizer.zero_grad();

            epoch_loss += loss;
            epoch_correct += correct;
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch
                      << ", BCE Loss: " << epoch_loss / total_data
                      << ", Accuracy: " << static_cast<float>(epoch_correct) / total_data
                      << std::endl;
            layer1.print_weight("Layer1");
            layer2.print_weight("Layer2");
            std::cout << "----------------------------------------" << std::endl;
        }
    }
}

int main() {
    CudaDeviceInfo::PrintAllDevices();
    train();
    return 0;
}