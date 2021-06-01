#include "kernels.cuh"
#include <iostream>


__global__ void matrixVectorV1(myFloat* A, myFloat* v, myFloat* res, unsigned int N) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = N * THREADS_NUMBER * by;
	int aEnd = aBegin + N - 1;
	int bBegin = THREADS_NUMBER;

	int aStep = THREADS_NUMBER;
	int bStep = THREADS_NUMBER;

	myFloat Csub = 0;

	for (int a = aBegin, b = bBegin; a <= aEnd;
		a += aStep, b += bStep) {
		__shared__ myFloat As[THREADS_NUMBER][THREADS_NUMBER];
		__shared__ myFloat Bs[THREADS_NUMBER];

		As[ty][tx] = A[a + N * ty + tx];
		Bs[tx] = B[b + tx + ty];

		__syncthreads();

		for (int k = 0; k < THREADS_NUMBER; ++k)
			Csub += As[ty][k] * Bs[k];

		__syncthreads();
	}

	int c = THREADS_NUMBER * by + THREADS_NUMBER * bx;
	res[c + ty + tx] = Csub;
}
