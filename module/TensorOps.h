#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <memory>
#include <cassert>  // 為了 assert
#include <ostream>
#include <iostream>
#include "Tensor.cuh"

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // std::cout << a->size() << " " << b->size() << std::endl;
    assert(a->batch_size == b->batch_size && a->features == b->features);
    auto result = std::make_shared<Tensor>(a->batch_size, a->features, a->requires_grad || b->requires_grad);

    for (int i = 0; i < result->size(); ++i) {
        result->data[i] = a->data[i] + b->data[i];
    }

    if (result->requires_grad) {
        if (a->requires_grad) {
            result->dependencies.emplace_back(a, [a](const float* grad_out) {
                for (int i = 0; i < a->size(); ++i)
                    a->grad[i] += grad_out[i];
            });
        }
        if (b->requires_grad) {
            result->dependencies.emplace_back(b, [b](const float* grad_out) {
                for (int i = 0; i < b->size(); ++i)
                    b->grad[i] += grad_out[i];
            });
        }
    }

    return result;
}

#endif // TENSOR_OPS_H
