#include "kernels.cuh"
#include <iostream>


__global__ void matrixAdd(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {

	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidX < N && tidY < N) {
		C[tidY * N + tidX] = A[tidY * N + tidX] + B[tidY * N + tidX];
	}
	
}

__global__ void matrixMulV2(myFloat* a, myFloat* b, myFloat* c, unsigned int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ myFloat s_a[THREADS_NUMBER];
	__shared__ myFloat s_b[THREADS_NUMBER];

	myFloat tmp = 0;

	for (int i = 0; i < N; i += blockDim.x) {
		s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
		s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

		__syncthreads();

		for (int j = 0; j < blockDim.x; j++) {
			tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
		}

		__syncthreads();
	}

	c[row * N + col] = tmp;
}

__global__ void matrixMulV3(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	myFloat tmp = 0;

	while(tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k++)
		{
			tmp += A[tidY + k] * B[k * N + tidX];
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

	float tmp = 0;

	while (tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k += 4)
		{
			float4 a_tmp = reinterpret_cast<float4*>(&A[tidY * N + k])[0];
			tmp += a_tmp.x * B[k * N + tidX];
			tmp += a_tmp.y * B[(k + 1) * N + tidX];
			tmp += a_tmp.z * B[(k + 2) * N + tidX];
			tmp += a_tmp.w * B[(k + 3) * N + tidX];
		}
		C[tidY * N + tidX] = tmp;
		tidX += strideX;
		tidY += strideY;
	}
}

__global__ void matrixMulV4d(double* A, double* B, double* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	double tmp = 0;

	while (tidX < N && tidY < N) {
		
		for (size_t k = 0; k < N; k += 4)
		{
			float4 a_tmp = reinterpret_cast<float4*>(&A[tidY * N + k])[0];
			tmp += a_tmp.x * B[k * N + tidX];
			tmp += a_tmp.y * B[(k + 1) * N + tidX];
			tmp += a_tmp.z * B[(k + 2) * N + tidX];
			tmp += a_tmp.w * B[(k + 3) * N + tidX];
		}
		C[tidY * N + tidX] = tmp;
		tidX += strideX;
		tidY += strideY;
	}
}

__global__ void matrixMulV4i(int* A, int* B, int* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	double tmp = 0;

	while (tidX < N && tidY < N) {

		for (size_t k = 0; k < N; k += 4)
		{
			int4 a_tmp = reinterpret_cast<int4*>(&A[tidY * N + k])[0];
			tmp += a_tmp.x * B[k * N + tidX];
			tmp += a_tmp.y * B[(k + 1) * N + tidX];
			tmp += a_tmp.z * B[(k + 2) * N + tidX];
			tmp += a_tmp.w * B[(k + 3) * N + tidX];
		}
		C[tidY * N + tidX] = tmp;
		tidX += strideX;
		tidY += strideY;
	}
}
