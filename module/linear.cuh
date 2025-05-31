#pragma once
#include "tensor.cuh"
#include <memory>
#include <string>
#include <vector>

class Linear {
  public:
    int in_features;
    int out_features;
    struct Tensor *bias;
    struct Tensor *weight;
    Linear(int in_f, int out_f);
    Tensor *forward(Tensor *input);
    Tensor *_tensor();
    void print_weight(const std::string &name = "") const;
    void print_grad(const std::string &name = "") const;
    float get_weight(int in_idx, int out_idx) const;
    ~Linear();
};
