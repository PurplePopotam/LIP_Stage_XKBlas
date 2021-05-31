#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include <cuda_runtime.h>
#include "matrix.hpp"

#define THREADS_NUMBER 32

__global__ void matrixVectorV1(myFloat* A, myFloat* v, unsigned int N);

#endif
