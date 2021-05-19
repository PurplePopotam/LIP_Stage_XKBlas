#include <stdio.h>
#include <cassert>
#include <iostream>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"


#define ITER 10

//Prise de temps CPU des kernels 
//Prise de temps séparemment des transfert de données
//Boucle pour mesurer sur plusieurs lancements
//

myFloat cpuDotProduct(myFloat* a, myFloat* b, int n) {
    myFloat res = 0.0f;

    for (int i = 0; i < n; i++) {
        res += a[i] * b[i];
    }

    return res;
}


int main(int argc, char** argv) {

    #define N atoi(argv[1])
    //CudaEvents are used to measure the execution time on the GPU
    bool debug = false;

    cudaEvent_t startGPU;
    cudaEvent_t stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float milliseconds;

    std::chrono::duration<double, std::milli> millisecondsCPUhost;
    std::chrono::duration<double, std::milli> millisecondsCPUdevice;
    std::chrono::duration<double, std::milli> millisecondsDeviceHostCopy;

    
    size_t bytes = sizeof(myFloat) * N;

    std::cout << "DotProduct of " << N/1000000 << "M elements, using " << THREADS_PER_BLOCK << " threads per block. " << std::endl;
    std::cout << std::endl;
    std::cout << "Iteration " << " | " << "CPU host exec time" << " | " << "GPU device exec time" << " | " << "CPU device exec time" << " | " << "device -> host copy duration" << std::endl;

    for (size_t i = 0; i < ITER; i++) {

        myFloat* h_a, * h_b, * h_c;
        myFloat* d_a, * d_b, * d_c;
        myFloat resCPU;

        h_a = new myFloat[N];
        h_b = new myFloat[N];
        h_c = new myFloat;

        cudaMalloc((void**)&d_a, bytes);
        cudaMalloc((void**)&d_b, bytes);
        cudaMalloc((void**)&d_c, sizeof(myFloat));

        for (size_t j = 0; j < N; j++) {
            h_a[j] = rand() % 10;
            h_b[j] = rand() % 10;
        }
        *h_c = 0;

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c, sizeof(myFloat), cudaMemcpyHostToDevice);

        int NUM_BLOCKS = N / THREADS_PER_BLOCK;

        //Doing the dot product on the host

        auto startCPUhost = std::chrono::high_resolution_clock::now();
        resCPU = cpuDotProduct(h_a, h_b, N);
        auto stopCPUhost = std::chrono::high_resolution_clock::now();
        
        millisecondsCPUhost = stopCPUhost - startCPUhost;

        //Doing the dot product on the device

        auto startCPUdevice = std::chrono::high_resolution_clock::now();
        cudaEventRecord(startGPU);

        dotProductV3 << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, N);

        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU);
        auto stopCPUdevice = std::chrono::high_resolution_clock::now();

        millisecondsCPUdevice = stopCPUdevice - startCPUdevice;

        auto startDeviceHostCopy = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_c, d_c, sizeof(myFloat), cudaMemcpyDeviceToHost);
        auto stopDeviceHostCopy = std::chrono::high_resolution_clock::now();

        millisecondsDeviceHostCopy = stopDeviceHostCopy - startDeviceHostCopy;

        cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);

        if (debug) {
            std::cout << "Dot product results on GPU : " << *h_c << std::endl;
            std::cout << "Dot product results on CPU : " << resCPU << std::endl;
            std::cout << std::endl;
        }

        //Results display

        
        std::cout << std::endl;
        std::cout << "     " << i << "    " << " |     " << millisecondsCPUhost.count() << " ms " << "    |       " << milliseconds << " ms " << "    |      " << millisecondsCPUdevice.count() << " ms " << "      |      " << millisecondsDeviceHostCopy.count() << " ms" << std::endl;

        cudaFree(&d_a); cudaFree(&d_b); cudaFree(&d_c);
        delete(h_a); delete(h_b); delete(h_c);
    }

    return 0;
}