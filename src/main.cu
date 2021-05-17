#include <stdio.h>
#include <cassert>
#include <iostream>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"


#define N 4000000
#define THREADS_PER_BLOCK 512
float cpuDotProduct(float* a, float* b, int n) {
    float res = 0.0f;

    for (int i = 0; i < n; i++) {
        res += a[i] * b[i];
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



    float* h_a, * h_b, * h_c;
    float* d_a, * d_b, * d_c;

    h_a = new float[N];
    h_b = new float[N];
    h_c = new float;

    size_t bytes = sizeof(float) * N;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, sizeof(float));

    for (size_t i = 0; i < N; i++) {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }
    *h_c = 0;

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(float), cudaMemcpyHostToDevice);

    int NUM_THREADS = 512;
    int NUM_BLOCKS = N / THREADS_PER_BLOCK;

    //Doing the dot product on the device
    cudaEventRecord(startGPU);
    dotProductV2 <<<NUM_BLOCKS, NUM_THREADS>>> (d_a, d_b, d_c, N);
    cudaEventRecord(stopGPU);

    cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopGPU);

    //Doing the dot product on the host
    auto startCPU = std::chrono::high_resolution_clock::now();
    float res = cpuDotProduct(h_a, h_b, N);
    auto stopCPU = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> millisecondsCPU = stopCPU - startCPU;

    cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);
    

    std::cout << "Dot product results on CPU : " << res << std::endl;
    std::cout << "Dot product results on GPU : " << *h_c << std::endl;
    std::cout << std::endl;
    std::cout << "GPU execution time : " << milliseconds << " ms . " << std::endl;
    //std::cout << "GPU bandwidth : " << N * 6 / milliseconds / 1e6 << " GB/s ." << std::endl;    //Not the accurate numbre of flops 
    std::cout << std::endl;
    std::cout << "CPU execution time : " << millisecondsCPU.count() << " ms ." << std::endl;    
    std::cout << "CPU bandwidth : " << N * 3 / millisecondsCPU.count() / 1e6 << " GB/s ." << std::endl; //3 W/R operations in the cpuDotProduct function

    std::cout << std::endl;
    return 0;
}