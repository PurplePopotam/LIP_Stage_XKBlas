#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__
#include <cuda_runtime.h>

__global__ void Product(float* a, float* b, float* c, int n);

__global__ void dotProductV1(float* a, float* b, float* c, unsigned int n);	//Sum reduction on 1 thread...

__global__ void dotProductV2(float* x, float* y, float* dot, unsigned int n);	//Youtube tuto 

__global__ void dotProductV3(float* x, float* y, float* dot, unsigned int n); //NVIDIA webinar2 slides, unroll last warp

__global__ void dotProductV4(float* x, float* y, float* dot, unsigned int n);	//Complete unroll using templates

__device__ void warpReduce(volatile float* sdata, unsigned int tid);

#endif