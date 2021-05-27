#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.cuh"

#define THREADS_NUMBER 32

__global__ void matrixAddV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N);

__global__ void matrixMulV3(myFloat* A, myFloat* B, myFloat* C, unsigned int N);	//Not tiles with a tmp variable instead of 2N accesses

__global__ void matrixMulV4f(float* A, float* B, float* C, unsigned int N);	//prefetch 4 items from A, float version

__global__ void matrixMulV4d(double* A, double* B, double* C, unsigned int N); //prefetch 4 items from A, double version


#endif
