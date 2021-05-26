#include "kernels.cuh"
#include <iostream>


__global__ void matrixAddV1(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {

	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidX < N && tidY < N) {
		C[tidY * N + tidX] = A[tidY * N + tidX] + B[tidY * N + tidX];
	}
	
}

__global__ void matrixMulV3(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	myFloat tmp = 0;

	while(tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k+=4)
		{
			float4
			tmp += A[tidY * N + k] * B[k * N + tidX];
		}
		C[tidY * N + tidX] = tmp;
		tidX += strideX;
		tidY += strideY;
	}
}

__global__ void matrixMulV4f(float* A, float* B, float* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	myFloat tmp = 0;

	while (tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k += 4)
		{
			float4 a_tmp = reinterpret_cast<float4*>(&a[row * N + i])[0];
			tmp += a_tmp.x * B[k * N + tidX];
			tmp += a_tmp.y * B[(k + 1) * N + tidX];
			tmp += a_tmp.z * B[(k + 1) * N + tidX];
			tmp += a_tmp.w * B[(k + 1) * N + tidX];
		}
		C[tidY * N + tidX] = tmp;
		tidX += strideX;
		tidY += strideY;
	}
}