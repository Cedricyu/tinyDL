#include "Linear.cuh"
#include <cassert>
#include <iostream>
#include <vector>

void test_linear() {
    const int batch_size = 2;
    const int in_features = 4;
    const int hidden_features = 5;
    const int out_features = 3;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; // shape: 2 x 4

    std::vector<float> hidden_output(batch_size * hidden_features);
    std::vector<float> final_output(batch_size * out_features);

    Linear linear1(in_features, hidden_features);
    Linear linear2(hidden_features, out_features);

    // Forward
    linear1.forward(input.data(), hidden_output.data(), batch_size);
    linear2.forward(hidden_output.data(), final_output.data(), batch_size);

    std::cout << "Final Output:" << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features; ++j) {
            std::cout << final_output[i * out_features + j] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> grad_output = {1.0f, 0.5f, -1.0f, -0.5f, 2.0f, 0.0f}; // shape: 2 x 3

    linear2.backward(hidden_output.data(), grad_output.data(), batch_size);
    linear1.backward(input.data(), hidden_output.data(), batch_size);

    std::cout << "Backward pass done for 2-layer network." << std::endl;
}

int main() {
    test_linear();

    // 你可以未來加更多：
    // test_relu();
    // test_step();
    // test_sequential();
}