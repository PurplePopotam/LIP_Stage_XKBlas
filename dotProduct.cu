#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void produitPasScalaire(float* a, float* b, float* c, int n) {

    float res;

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }

}

int main() {
    int N = 1 << 16;

    float* h_a, * h_b, * h_c;
    float* d_a, * d_b, * d_c;

    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];

    size_t bytes = sizeof(float) * N;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for (size_t i = 0; i < N; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
        h_c[i] = 0;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

    produitPasScalaire <<<NUM_BLOCKS, NUM_THREADS>>> (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    std::cout << "finished succesfully" << std::endl;

    for (size_t i = 0; i < 10; i++) {
        std::cout << h_a[i] << " * " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    std::cout << std::endl;

    return 0;
}