// module/Tensor.cuh
#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

class Tensor;

struct Dependency {
    std::shared_ptr<Tensor> tensor;
    std::function<void(const float *grad_out)> backward_fn;

    Dependency(std::shared_ptr<Tensor> t, std::function<void(const float *)> fn)
        : tensor(std::move(t)), backward_fn(std::move(fn)) {
    }
};

class Tensor : public std::enable_shared_from_this<Tensor> {
  public:
    float *data;
    float *grad;
    int batch_size, features;
    bool requires_grad;
    std::vector<Dependency> dependencies;

    Tensor(int batch, int feat, bool requires_grad = false)
        : batch_size(batch), features(feat), requires_grad(requires_grad) {
        data = new float[batch * feat]();
        grad = requires_grad ? new float[batch * feat]() : nullptr;
    }

    // 外部指標版
    Tensor(float *external_data, int batch, int feat)
        : data(external_data), batch_size(batch), features(feat), requires_grad(false), grad(nullptr) {
    }

    int size() const {
        return batch_size * features;
    }

    void zero_grad() {
        if (requires_grad && grad) {
            std::fill(grad, grad + size(), 0.0f);
        }
    }

    void backward() {
        if (!requires_grad)
            return;
        std::fill(grad, grad + size(), 1.0f);
        for (auto &dep : dependencies) {
            dep.backward_fn(grad);
        }
    }

    std::shared_ptr<Tensor> shared() {
        return shared_from_this();
    }

    ~Tensor() {
        delete[] data;
        if (grad)
            delete[] grad;
    }
};

#endif // TENSOR_H
