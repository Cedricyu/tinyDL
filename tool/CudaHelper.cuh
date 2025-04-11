#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

void checkCudaError(cudaError_t err, const std::string &msg);

#endif // CUDA_HELPER_H
