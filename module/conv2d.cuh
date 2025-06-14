#ifndef CONV2D_CUH
#define CONV2D_CUH

#include <stdio.h>
#include "tensor.cuh"

Tensor *conv2d_forward(Tensor *x, Tensor *w, int padding, int stride);


class Conv2D {
  private:
    Tensor *w;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int padding_h, padding_w;

  public:
    Conv2D(Tensor *weight, int kh, int kw, int ph = 0, int pw = 0, int sh = 1, int sw = 1)
        : w(weight), kernel_h(kh), kernel_w(kw), padding_h(ph), padding_w(pw), stride_h(sh), stride_w(sw) {}

    Tensor *forward(Tensor *x) { return conv2d_forward(x, w, padding_h, stride_h); }

    void print_weight() {
        if (w) {
            tensor_print(w);
        } else {
            printf("No weight initialized.\n");
        }
    }

    void print_grad() {
        if (w && w->grad) {
            tensor_print_grad(w);
        } else {
            printf("No gradient available.\n");
        }
    }

    ~Conv2D() {
        // if (w) tensor_free(w);
    }
};

#endif  // CONV2D_CUH
