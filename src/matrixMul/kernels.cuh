#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.hpp"

#define THREADS_NUMBER 16

__global__ void matrixAdd(myFloat* A, myFloat* B, myFloat* C, unsigned int N);

__global__ void matrixMulV2(myFloat* a, myFloat* b, myFloat* c, unsigned int N);	//tiled version

__global__ void matrixMulV3(myFloat* A, myFloat* B, myFloat* C, unsigned int N);	//Not tiled, with a tmp variable instead of 2N accesses

__global__ void matrixMulV4f(float* A, float* B, float* C, unsigned int N);	//Not tiled, prefetch 4 items from A, float version

__global__ void matrixMulV4d(double* A, double* B, double* C, unsigned int N); //Not tiled, prefetch 4 items from A, double version

__global__ void matrixMulV4i(int* A, int* B, int* C, unsigned int N); //Not tiled, prefetch 4 items from A, int version

__global__ void matrixVectorV1(myFloat* A, myFloat* v, unsigned int N);
#endif
