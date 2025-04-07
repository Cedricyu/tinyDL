#pragma once
#include <string>
#include <vector>
#include <memory>     
#include "Tensor.cuh" 
#include <cublas_v2.h>

class Linear {
  public:
    float *d_weight;
    float *d_bias;
    float *d_grad_weight;
    float *d_grad_bias;
    int in_features, out_features;

    Linear(int in_f, int out_f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> input,
                                     std::shared_ptr<Tensor> grad_output);
    void print_weight(const std::string &name = "") const;
    float get_weight(int in_idx, int out_idx) const;
    ~Linear();
};
