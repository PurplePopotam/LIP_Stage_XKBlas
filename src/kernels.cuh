#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__
#include <cuda_runtime.h>

__global__ void Product(float* a, float* b, float* c, int n);

__global__ void dotProductV1(float* a, float* b, float* c, unsigned int n);

__global__ void dotProductV2(float* x, float* y, float* dot, unsigned int n);

#endif