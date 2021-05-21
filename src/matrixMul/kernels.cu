#include "kernels.cuh"
#include <iostream>

__global__ void matrixAddV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {

	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	//unsigned int strideX = blockDim.x * gridDim.x;
	//unsigned int strideY = blockDim.y * gridDim.y;

	if (tidX < N && tidY < N) {
		C[tidY * N + tidX] = A[tidY * N + tidX] + B[tidY * N + tidX];
	}
	
}

__global__ void matrixMulV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k++)
		{
			C[tidY * N + tidX] +=  A[k * N + tidX] * B[tidY * N + k];
		}
	}
}

__global__ void matrixMulV2(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;

	__shared__ myFloat cache_A[THREADS_NUMBER];
	__shared__ myFloat cache_B[THREADS_NUMBER];

	myFloat tmp = 0;


}