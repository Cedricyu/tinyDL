#include "CudaDeviceInfo.cuh"
#include "test.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

int main() {
    CudaDeviceInfo::PrintAllDevices();

    std::cout << "Running tests..." << std::endl;

    test_linear();

    // 你可以未來加更多：
    // test_relu();
    // test_step();
    // test_sequential();

    return 0;
}