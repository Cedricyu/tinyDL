#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Max Grid Dimensions: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
    }

    return 0;
}
