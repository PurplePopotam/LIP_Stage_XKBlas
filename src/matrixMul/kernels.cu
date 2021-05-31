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
	
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = N * THREADS_NUMBER * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + N - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = THREADS_NUMBER;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = THREADS_NUMBER * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = THREADS_NUMBER * N;

	myFloat Csub = 0;

	for (int a = aBegin, b = bBegin;a <= aEnd; a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ myFloat As[THREADS_NUMBER][THREADS_NUMBER];
		// Shared memory for the sub-matrix of B
		__shared__ myFloat Bs[THREADS_NUMBER][THREADS_NUMBER];
		// Load the matrices from global memory to shared memory;
		// each thread loads one element of each matrix
		As[ty][tx] = A[a + N * ty + tx];
		Bs[ty][tx] = B[b + N * ty + tx];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();
		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < THREADS_NUMBER; ++k)
			Csub += As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	int c = N * THREADS_NUMBER * by + THREADS_NUMBER * bx;
	C[c + N * ty + tx] = Csub;
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
