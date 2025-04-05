#pragma once

class SGD {
  public:
    SGD(float learning_rate);
    void add_param(float *param, float *grad, int size);
    void step();
    void zero_grad();
};
