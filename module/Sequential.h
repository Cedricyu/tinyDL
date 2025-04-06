#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "Linear.cuh"
#include "SGD.cuh"
#include <vector>
#include <cassert>

class Sequential {
public:
    std::vector<Linear*> layers;
    std::vector<std::vector<float>> intermediates; // intermediate activations

    void add(Linear* layer) {
        layers.push_back(layer);
    }

    void register_parameters(SGD& optimizer) {
        for (auto layer : layers) {
            optimizer.add_param(layer->d_weight, layer->d_grad_weight, layer->in_features * layer->out_features);
            optimizer.add_param(layer->d_bias, layer->d_grad_bias, layer->out_features);
        }
    }

    void forward(float* input, float* output, int batch_size) {
        intermediates.clear();
        intermediates.emplace_back(input, input + batch_size * layers[0]->in_features); // store input

        float* current_input = input;

        for (size_t i = 0; i < layers.size(); ++i) {
            int out_dim = layers[i]->out_features;
            std::vector<float> temp(batch_size * out_dim);
            layers[i]->forward(current_input, temp.data(), batch_size);
            intermediates.push_back(temp); // store activation
            current_input = intermediates.back().data(); // for next layer
        }

        // copy last output to user-supplied buffer
        std::copy(intermediates.back().begin(), intermediates.back().end(), output);
    }

    std::vector<float> backward(float* grad_output, int batch_size) {
        float* current_grad = grad_output;

        for (int i = layers.size() - 1; i >= 0; --i) {
            float* input = intermediates[i].data(); // input to current layer
            std::vector<float> grad_input = layers[i]->backward(input, current_grad, batch_size);

            if (i == 0)
                return grad_input; // final grad_input returned to caller
            current_grad = grad_input.data(); // for next layer
        }

        // should not reach here
        assert(false && "backward logic error");
        return {};
    }

    ~Sequential() {
        for (auto layer : layers) delete layer;
    }
};

#endif
