#include "CudaKernels.cuh"
#include "Linear.cuh"
#include "Tensor.cuh"
#include "TensorOps.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>

Linear::Linear(int in_f, int out_f) {
    in_features = in_f;
    out_features = out_f;

    weight = tensor_create(in_f, out_f, 1);  // (K, N)
    fill_tensor_with_random(weight);
    bias = tensor_create(1, out_f, 1);       // (1, N)
}

Tensor *Linear::forward(Tensor *input) {
    Tensor *out = tensor_matmul(input, this->weight); // z = x × y
    // out = tensor_add_bias(out, this->bias); // z = x × y + b
    return out;
}

Linear::~Linear() {
    tensor_free(this->weight);
    tensor_free(this->bias);
}

void Linear::print_weight(const std::string &name) const{
    if (name.empty()) {
        std::cout << "Weight: " << std::endl;
    } else {
        std::cout << name << ": " << std::endl;
    }
    tensor_print(this->weight);
}

void Linear::print_grad(const std::string &name) const{
    if (name.empty()) {
        std::cout << "Grad: " << std::endl;
    } else {
        std::cout << name << ": " << std::endl;
    }
    tensor_print_grad(this->weight);
}