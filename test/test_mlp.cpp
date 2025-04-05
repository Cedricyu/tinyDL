#include <iostream>
#include <vector>
#include <cassert>
#include <numeric> // For std::iota, std::fill
#include "MLP.cuh"

void test_mlp() {
    std::cout << "\n--- Testing MLP  ---\n" << std::endl;

    const int batch_size = 2;
    const std::vector<int> layer_sizes = {3, 4, 4, 3};
    const int input_features = layer_sizes.front();
    const int output_features = layer_sizes.back();

    MLP mlp(layer_sizes, batch_size);
    std::cout << "MLP created with " << mlp.num_layers << " layers." << std::endl;

    std::vector<float> h_input(batch_size * input_features);
    std::iota(h_input.begin(), h_input.end(), 1.0f); // Fill with 1.0, 2.0, ...

    std::vector<float> h_grad_output(batch_size * output_features);
    std::fill(h_grad_output.begin(), h_grad_output.end(), 0.5f); // Fill with 0.5

    std::cout << "MLP forward..." << std::endl;
    // mlp.forward(); // This needs modification once MLP::forward is implemented
                     // to accept input data (e.g., mlp.forward(h_input.data()))
                     // and potentially return output.

    std::cout << "MLP backward..." << std::endl;
    // mlp.backward(); // This needs modification once MLP::backward is implemented
                      // to accept output gradients (e.g., mlp.backward(h_grad_output.data()))

    std::cout << "MLP pass done for " << mlp.num_layers << "-layer network." << std::endl;
}
