#include "test.h"
#include "CudaDeviceInfo.h"

int main() {
    CudaDeviceInfo::PrintAllDevices();

    test_linear();
    
    // 你可以未來加更多：
    // test_relu();
    // test_step();
    // test_sequential();

    return 0;
}