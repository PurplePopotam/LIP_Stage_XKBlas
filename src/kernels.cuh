#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

__global__ void Product(float* a, float* b, float* c, int n);

__global__ void dotProductV1(float* a, float* b, float* c, unsigned int n);	//Sum reduction on 1 thread...

__global__ void dotProductV2(float* x, float* y, float* dot, unsigned int n);	//Youtube tuto 

__global__ void dotProductV3(float* x, float* y, float* dot, unsigned int n); //NVIDIA webinar2 slides, unroll last warp

__device__ void warpReduce(volatile float* sdata, unsigned int tid);

template <unsigned int blockSize>
__device__ void warpReduceT(volatile float* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

//Complete unroll using templates

template <unsigned int blockSize>
__global__ void dotProductV4(float* x, float* y, float* dot, unsigned int n) {	
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

	if (blockSize >= 512) {
		if (tid < 256) {
			cache[tid] = cache[tid + 256];	
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			cache[tid] = cache[tid + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			cache[tid] = cache[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduceT<blockSize>(cache, tid);
	}

	if (tid == 0) {
		atomicAdd(dot, cache[0]);
	}
}	


#endif