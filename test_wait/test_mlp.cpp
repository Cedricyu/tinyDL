#include "MLP.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric> // For std::iota, std::fill
#include <vector>

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
    auto input_tensor = std::make_shared<Tensor>(batch_size, input_features);
    cudaMemcpy(input_tensor->data, h_input.data(), batch_size * input_features * sizeof(float), cudaMemcpyHostToDevice);
    auto output_tensor = mlp.forward(input_tensor);

    std::cout << "MLP backward..." << std::endl;
    auto grad_output_tensor = std::make_shared<Tensor>(batch_size, output_features);
    cudaMemcpy(grad_output_tensor->data,
               h_grad_output.data(),
               batch_size * output_features * sizeof(float),
               cudaMemcpyHostToDevice);
    mlp.backward(input_tensor, grad_output_tensor);

    std::cout << "MLP pass done for " << mlp.num_layers << "-layer network." << std::endl;
}
