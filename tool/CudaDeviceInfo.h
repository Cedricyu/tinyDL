#ifndef CUDA_DEVICE_INFO_H
#define CUDA_DEVICE_INFO_H

#include <string>
#include <vector>

struct DeviceInfo {
    int deviceId;
    std::string name;
    int maxGridDimX;
    int maxGridDimY;
    int maxGridDimZ;
    int maxThreadsPerBlock;
    int maxThreadsPerSM;
    size_t totalGlobalMemMB;
    int numSMs;
};

class CudaDeviceInfo {
  public:
    static std::vector<DeviceInfo> GetAllDevices();
    static void PrintAllDevices();
};

#endif // CUDA_DEVICE_INFO_H
