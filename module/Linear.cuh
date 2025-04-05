#pragma once
#include <string>
#include <vector>

class Linear {
  public:
    float *d_weight;
    float *d_bias;
    float *d_grad_weight;
    float *d_grad_bias;
    int in_features, out_features;

    Linear(int in_f, int out_f);
    void forward(float *input, float *output, int batch_size);
    std::vector<float> backward(float *input, float *grad_output, int batch_size);
    void print_weight(const std::string &name = "") const;
    float get_weight(int in_idx, int out_idx) const;
    ~Linear();
};
