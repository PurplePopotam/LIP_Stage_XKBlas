#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.cuh"

__global__ void matrixAddV1(myFloat* , myFloat* B, myFloat* C, unsigned int N);


#endif