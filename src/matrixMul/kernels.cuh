#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.cuh"

#define THREADS_NUMBER 16

__global__ void matrixAddV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N);

__global__ void matrixMulV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N);	//Naive implementation	

__global__ void matrixMulV2(myFloat* A, myFloat* B, myFloat* C, unsigned int N);	//Cache-tiled version, no coaelesced access

#endif