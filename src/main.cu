#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"

//Test push to 2 remotes
// 
//At the moment the products needed for vector dot product are computed on the device, 
//however the sum is computed on the host...

float sum(float* a, int n) {
    float res = 0.0f;

    for (int i = 0; i < n; i++) {
        res += a[i];
    }

    return res;
}


int main() {

    //CudaEvents are used to measure the execution time on the GPU
    cudaEvent_t startGPU;
    cudaEvent_t stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float milliseconds;

    int N = 20000000;

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
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
        h_c[i] = 0;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

    //Doing products on the device
    cudaEventRecord(startGPU);
    dotProduct <<<NUM_BLOCKS, NUM_THREADS>>> (d_a, d_b, d_c, N);
    cudaEventRecord(stopGPU);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopGPU);

    //Doing the sum on the host
    auto startCPU = std::chrono::high_resolution_clock::now();
    float res = sum(h_c, N);
    auto stopCPU = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> millisecondsCPU = stopCPU - startCPU;

    cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);
    

    std::cout << "Dot product results : " << res << std::endl;

    std::cout << "GPU execution time : " << milliseconds << " ms . " << std::endl;
    std::cout << "GPU bandwidth : " << N * 3 / milliseconds / 1e6 << " GB/s ." << std::endl;    //Only 3 W/R operations in the dotProduct kernel
    std::cout << "CPU execution time : " << millisecondsCPU.count() << " ms ." << std::endl;    //Only 2 W/R operations in the sum function
    std::cout << "CPU bandwidth : " << N * 2 / millisecondsCPU.count() / 1e6 << " GB/s ." << std::endl;
    std::cout << std::endl;
    return 0;
}