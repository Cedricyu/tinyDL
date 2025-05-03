#ifndef FLOAT_PRECISION_H
#define FLOAT_PRECISION_H

// 選擇一種訓練精度：UNCOMMENT ONE
//#define FLOAT_TYPE_FP8
//#define FLOAT_TYPE_FP16
#define FLOAT_TYPE_FP32

#ifdef FLOAT_TYPE_FP8
// 訓練用 float8（模擬用）
// CUDA 尚不原生支援 float8，可自定 8-bit 壓縮型別作為 placeholder
typedef uint8_t float_t;
#define FLOAT_EPSILON 0.05f

#elif defined(FLOAT_TYPE_FP16)
// CUDA 原生 half 精度（需要 __half）
#include <cuda_fp16.h>
typedef __half float_t;
#define FLOAT_EPSILON 1e-2f

#elif defined(FLOAT_TYPE_FP32)
typedef float float_t;
#define FLOAT_EPSILON 1e-6f

#else
#error "Must define one of: FLOAT_TYPE_FP8, FLOAT_TYPE_FP16, FLOAT_TYPE_FP32"
#endif

#endif // FLOAT_PRECISION_H
