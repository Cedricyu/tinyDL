#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "cudakernels.cuh"
#include "linear.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"

void xavier_init(Tensor *t, int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    for (int i = 0; i < t->batch_size * t->features; ++i) {
        float r = static_cast<float>(std::rand()) / RAND_MAX;  // [0,1]
        r = r * 2.0f * limit - limit;  // [ -limit, +limit ]
        t->data[i] = r;
    }
}

Linear::Linear(int in_f, int out_f) {
    in_features = in_f;
    out_features = out_f;

    weight = tensor_create(in_f, out_f, 1);  // (K, N)
    xavier_init(this->weight, in_features, out_features);        // 新增 Xavier 初始化
    bias = tensor_create(1, out_f, 1);  // (1, N)
}

Tensor *Linear::forward(Tensor *input) {
    Tensor *out = tensor_matmul(input, this->weight);  // z = x × y
    // out = tensor_add_bias(out, this->bias); // z = x × y + b
    return out;
}

Tensor *Linear::_tensor() { return this->weight; }

Linear::~Linear() {
    tensor_free(this->weight);
    tensor_free(this->bias);
}

void Linear::print_weight(const std::string &name) const {
    if (name.empty()) {
        std::cout << "Weight: " << std::endl;
    } else {
        std::cout << name << ": " << std::endl;
    }
    tensor_print(this->weight);
}

void Linear::print_grad(const std::string &name) const {
    if (name.empty()) {
        std::cout << "Grad: " << std::endl;
    } else {
        std::cout << name << ": " << std::endl;
    }
    tensor_print_grad(this->weight);
}