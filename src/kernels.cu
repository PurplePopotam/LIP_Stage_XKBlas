#include "kernels.cuh"
#include "stdio.h"

__global__ void dotProduct(float* a, float* b, float* c, int n) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}