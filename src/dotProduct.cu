#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

//Pour le moment la partie multiplication du produit scalaire est effectuée sur device, 
//mais la somme est encore sur l'hote...

__global__ void produitPasScalaire(float* a, float* b, float* c, int n) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

float sum(float* a, int n) {
    float res = 0.0f;

    for (int i = 0; i < n; i++) {
        res += a[i];
    }

    return res;
}


int main() {

    //Evènement CUDA pour mesurer le temps d'execution sur le GPU
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

    //On fait les multiplication sur le GPU
    cudaEventRecord(startGPU);
    produitPasScalaire <<<NUM_BLOCKS, NUM_THREADS>>> (d_a, d_b, d_c, N);
    cudaEventRecord(stopGPU);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopGPU);

    //On fait la somme sur l'hote
    auto startCPU = std::chrono::high_resolution_clock::now();
    float res = sum(h_c, N);
    auto stopCPU = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> millisecondsCPU = stopCPU - startCPU;

    cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);
    

    std::cout << "Resultat du produit scalaire : " << res << std::endl;

    std::cout << "Temps d'execution GPU : " << milliseconds << " ms . " << std::endl;

    std::cout << "Temps d'execution CPU : " << millisecondsCPU.count() << " ms ." << std::endl;

    std::cout << std::endl;
    return 0;
}