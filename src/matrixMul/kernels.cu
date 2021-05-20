#include "kernels.cuh"
#include <iostream>

__global__ void matrixAddV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {

	unsigned int tidX = threadIdx.x;
	unsigned int tidY = threadIdx.y;

	C[tidX * N + tidY] = A[tidX * N + tidY] + B[tidX * N + tidY];
}