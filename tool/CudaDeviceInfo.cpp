#include "CudaDeviceInfo.h"
#include <cuda_runtime.h>
#include <iostream>

std::vector<DeviceInfo> CudaDeviceInfo::GetAllDevices() {
    std::vector<DeviceInfo> devices;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        DeviceInfo info;
        info.deviceId = device;
        info.name = prop.name;
        info.maxGridDimX = prop.maxGridSize[0];
        info.maxGridDimY = prop.maxGridSize[1];
        info.maxGridDimZ = prop.maxGridSize[2];
        info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        info.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        info.totalGlobalMemMB = prop.totalGlobalMem / (1024 * 1024);
        info.numSMs = prop.multiProcessorCount;

        devices.push_back(info);
    }

    return devices;
}

void CudaDeviceInfo::PrintAllDevices() {
    auto devices = GetAllDevices();
    for (const auto &d : devices) {
        std::cout << "Device " << d.deviceId << ": " << d.name << std::endl;
        std::cout << "  Max Grid Dimensions: " << d.maxGridDimX << " x " << d.maxGridDimY << " x "
                  << d.maxGridDimZ << std::endl;
        std::cout << "  Max Threads per Block: " << d.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per SM: " << d.maxThreadsPerSM << std::endl;
        std::cout << "  Total Global Memory: " << d.totalGlobalMemMB << " MB" << std::endl;
        std::cout << "  Number of SMs: " << d.numSMs << std::endl;
    }
}
