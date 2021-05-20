#include "kernels.cuh"
#include <iostream>

__global__ void matrixAddV1(Matrix* A, Matrix* B, Matrix* C, unsigned int N) {
	unsigned int tidX = threadIdx.x;
	unsigned int tidY = threadIdx.y;

	C->content[tidX * N + tidY] = 1;

	 
}