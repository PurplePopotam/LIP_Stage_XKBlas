#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.cuh"

#define THREADS_NUMBER 16

__global__ void matrixAdd(myFloat* A, myFloat* B, myFloat* C, unsigned int N);

__global__ void matrixMulV2(const myFloat* a, const myFloat* b, myFloat* c);	//tiled version

__global__ void matrixMulV3(const myFloat* A, const myFloat* B, myFloat* C, unsigned int N);	//Not tiled, with a tmp variable instead of 2N accesses

__global__ void matrixMulV4f(const float* A, const float* B, float* C, unsigned int N);	//Not tiled, prefetch 4 items from A, float version

__global__ void matrixMulV4d(const double* A, const double* B, double* C, unsigned int N); //Not tiled, prefetch 4 items from A, double version

__global__ void matrixMulV4i(const int* A, const int* B, int* C, unsigned int N); //Not tiled, prefetch 4 items from A, int version

#endif
