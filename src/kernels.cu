#include "kernels.cuh"
#include "stdio.h"

#define THREADS_PER_BLOCK 512

__global__ void Product(float* a, float* b, float* c, int n) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

__global__ void dotProduct(float* a, float* b, float* c, int n) {


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