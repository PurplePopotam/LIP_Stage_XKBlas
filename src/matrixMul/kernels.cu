#include "kernels.cuh"
#include <iostream>

__global__ void matrixAddV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {

	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	if (tidX < N && tidY < N) {
		C[tidX * N + tidY] = A[tidX * N + tidY] + B[tidX * N + tidY];
	}
	
}