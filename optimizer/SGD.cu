#include "SGD.cuh"
#include <cuda_runtime.h>
#include <vector>

class SGDParam {
  public:
    float *param;
    float *grad;
    int size;
    SGDParam(float *p, float *g, int s) : param(p), grad(g), size(s) {
    }
};

__global__ void
sgd_update_kernel(float *param, float *grad, float lr, int size, float weight_decay) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        grad[idx] += weight_decay * param[idx];
        param[idx] -= lr * grad[idx];
    }
}

static std::vector<SGDParam> params;
static float global_lr;

SGD::SGD(float learning_rate) {
    global_lr = learning_rate;
}

void SGD::add_param(float *param, float *grad, int size) {
    params.emplace_back(param, grad, size);
}

void SGD::step() {
    float weight_decay = .0f; // default value

    for (auto &p : params) {
        int threads = 256;
        int blocks = (p.size + threads - 1) / threads;
        sgd_update_kernel<<<blocks, threads>>>(p.param, p.grad, global_lr, p.size, weight_decay);
    }
}

void SGD::zero_grad() {
    for (auto &p : params) {
        cudaMemset(p.grad, 0, p.size * sizeof(float));
    }
}
