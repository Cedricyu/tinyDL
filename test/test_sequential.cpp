#include "CudaDeviceInfo.h"
#include "Linear.cuh"
#include "SGD.cuh"
#include "Sequential.h"
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

void train_sequential(int epochs = 1000, int batch_size = 128) {
    const int input_dim = 2;
    const int hidden_dim = 8;
    const int output_dim = 1;

    const int total_data = 1024;
    std::vector<float> all_inputs(total_data * input_dim);
    std::vector<int> all_targets(total_data);
    generate_data_classification(all_inputs.data(), all_targets.data(), total_data);

    std::vector<float> inputs(batch_size * input_dim);
    std::vector<int> targets(batch_size);
    std::vector<float> outputs(batch_size * output_dim);
    std::vector<float> grad_output(batch_size * output_dim);
    std::vector<float> grad_input(batch_size * input_dim);

    // Build model
    Sequential model;
    model.add(new Linear(input_dim, hidden_dim));
    model.add(new Linear(hidden_dim, output_dim));

    SGD optimizer(0.0001f);
    model.register_parameters(optimizer);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        for (int i = 0; i < total_data; i += batch_size) {
            for (int j = 0; j < batch_size; ++j) {
                int idx = i + j;
                inputs[j * 2 + 0] = all_inputs[idx * 2 + 0];
                inputs[j * 2 + 1] = all_inputs[idx * 2 + 1];
                targets[j] = all_targets[idx];
            }

            model.forward(inputs.data(), outputs.data(), batch_size);

            float loss = 0.0f;
            int correct = 0;
            float eps = 1e-7f;

            for (int j = 0; j < batch_size; ++j) {
                float raw = outputs[j];
                float prob = sigmoid(raw);
                float label = static_cast<float>(targets[j]);

                loss += -(label * std::log(prob + eps) + (1 - label) * std::log(1 - prob + eps));
                grad_output[j] = prob - label;

                int pred = (prob > 0.5f) ? 1 : 0;
                if (pred == targets[j])
                    correct++;
            }

            std::vector<float> grad_input = model.backward(grad_output.data(), batch_size);

            optimizer.step();
            epoch_loss += loss;
            epoch_correct += correct;
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << ", BCE Loss: " << epoch_loss / total_data
                      << ", Accuracy: " << static_cast<float>(epoch_correct) / total_data << std::endl;

            // 若需要印出權重，可手動存取 layer
            model.layers[0]->print_weight();
            model.layers[1]->print_weight();
            std::cout << "----------------------------------------" << std::endl;
        }

        optimizer.zero_grad();
    }
}
