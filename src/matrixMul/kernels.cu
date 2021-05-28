#include "kernels.cuh"
#include <iostream>


__global__ void matrixAdd(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {

	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidX < N && tidY < N) {
		C[tidY * N + tidX] = A[tidY * N + tidX] + B[tidY * N + tidX];
	}
	
}

__global__ void matrixMulV2(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	
	__shared__ myFloat ds_A[THREADS_NUMBER][THREADS_NUMBER];
	__shared__ myFloat ds_B[THREADS_NUMBER][THREADS_NUMBER];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	myFloat Cvalue = 0;

	for(size_t t = 0; t < N/THREADS_NUMBER; t++){
		ds_A[ty][tx] = A[row * N + t * THREADS_NUMBER + tx];
		ds_B[ty][tx] = B[(t * THREADS_NUMBER + ty) * N + col];
		
		__syncthreads();

		for(size_t i = 0; i < THREADS_NUMBER; i++){
			Cvalue += ds_A[ty][i] * ds_B[i][tx];
		}

		__syncthreads();
	}
	C[row * N + col] = Cvalue;
}

__global__ void matrixMulV3(myFloat* A, myFloat* B, myFloat* C, unsigned int N) {
	unsigned int tidX = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tidY = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int strideX = blockDim.x * gridDim.x;
	unsigned int strideY = blockDim.y * gridDim.y;

	myFloat tmp = 0;

	if(tidX < N && tidY < N) {
		for (size_t k = 0; k < N; k++)
		{
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

	float tmp = 0;

	while(tidX < N && tidY < N) {
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
