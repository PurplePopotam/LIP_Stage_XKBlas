#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.cuh"

#define GRID_DIM 1

__global__ void matrixAddV1(myFloat* , myFloat* B, myFloat* C, unsigned int N);


#endif