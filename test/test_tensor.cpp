#include "TensorOps.h"
#include <iostream>

int main() {
    auto x = std::make_shared<Tensor>(1, 1, true);
    auto y = std::make_shared<Tensor>(1, 1, true);

    x->data[0] = 2.0f;
    y->data[0] = 3.0f;

    auto z = add(x, y);  // z = x + y
    z->backward();       // dz/dx = 1, dz/dy = 1

    std::cout << "x.grad = " << x->grad[0] << std::endl;
    std::cout << "y.grad = " << y->grad[0] << std::endl;

    return 0;
}
