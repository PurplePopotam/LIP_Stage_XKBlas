#include "kernels.cuh"
#include "stdio.h"

#define THREADS_PER_BLOCK 512

__global__ void Product(float* a, float* b, float* c, int n) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

__device__ void warpReduce(float* sdata, unsigned int tid) {

	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}


__global__ void dotProductV1(float* a, float* b, float* c, unsigned int n) {

    __shared__ float temp[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[tid] * b[tid];

    __syncthreads();

    if (threadIdx.x == 0) {
        float res = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            res += temp[i];
        }
        atomicAdd(c, res);
    }
}

__global__ void dotProductV2(float* x, float* y, float* dot, unsigned int n) {

	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	__shared__ float cache[THREADS_PER_BLOCK];

	double temp = 0.0;
	while (index < n) {
		temp += x[index] * y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	unsigned int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}


	if (threadIdx.x == 0) {
		atomicAdd(dot, cache[0]);
	}
}

__global__ void dotProductV3(float* x, float* y, float* dot, unsigned int n) {

	__shared__ float cache[THREADS_PER_BLOCK];

	unsigned int tid = threadIdx.x;	
	unsigned int index = blockIdx.x * blockDim.x + tid;	

	double temp = 0.0;
	while (index < n) {
		temp += x[index] * y[index];
		index += blockDim.x * gridDim.x;
	}

	cache[tid] = temp;

	__syncthreads();

	//Reduction

	for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
		if (tid < s) {
			cache[tid] += cache[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(cache, tid);
	}

	if (tid == 0) {
		atomicAdd(dot, cache[0]);
	}
}