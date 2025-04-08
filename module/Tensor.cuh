// Tensor.h
#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
  public:
    float *data;
    int batch_size;
    int features;

    Tensor(int batch, int feat) : batch_size(batch), features(feat) {
        data = new float[batch * feat]();
    }

    Tensor(float *external, int batch, int feat) : data(external), batch_size(batch), features(feat) {
    }

    int size() const {
        return batch_size * features;
    }

    ~Tensor() {
        delete[] data;
    }
};

#endif
