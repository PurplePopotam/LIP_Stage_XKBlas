#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.cuh"

#define BLOCK_DIM  (2,2)
#define GRID_DIM 1

__global__ void matrixAddV1(Matrix* A, Matrix* B, Matrix* C, unsigned int N);


#endif