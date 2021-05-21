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
	//global IDs
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k++)
		{
			C[tidY * N + tidX] +=  A[tidY * N + k] * B[k * N + tidX];
		}
	}
}

__global__ void matrixMulV2(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	__shared__ myFloat cache_A[THREADS_NUMBER * THREADS_NUMBER];
	__shared__ myFloat cache_B[THREADS_NUMBER * THREADS_NUMBER];

	unsigned int tidX = threadIdx.x + THREADS_NUMBER * blockIdx.x;
	unsigned int tidY = threadIdx.y + THREADS_NUMBER * blockIdx.y;

	myFloat tmp = 0;

	for (size_t i = 0; i < N/THREADS_NUMBER; ++i)
	{
		cache_A[threadIdx.y * THREADS_NUMBER + threadIdx.x] = A[tidY * N + (i * THREADS_NUMBER + threadIdx.x)];
		cache_B[threadIdx.y * THREADS_NUMBER + threadIdx.x] = B[(i * THREADS_NUMBER + threadIdx.y) * N + tidX];
		__syncthreads();

		for (size_t k = 0; k < THREADS_NUMBER; ++k)
		{
			tmp += cache_A[threadIdx.y * THREADS_NUMBER + k] * cache_B[k * THREADS_NUMBER + threadIdx.x];
		}
	}
	C[tidY * N + tidX] = tmp;
}
