#include <CudaHelper.cuh>
#include <iostream>

void checkCudaError(cudaError_t err, const std::string &msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}