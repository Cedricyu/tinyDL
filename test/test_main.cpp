#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "CudaDeviceInfo.cuh"
#include "test.h"

int main() {
    CudaDeviceInfo::PrintAllDevices();

    std::cout << "Running tests..." << std::endl;

    // test_linear();
    test_conv2d();
    // 你可以未來加更多：
    // test_relu();
    // test_step();
    // test_sequential();

    return 0;
}