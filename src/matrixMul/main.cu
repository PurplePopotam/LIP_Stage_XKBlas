#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"
#include <iostream>


int main(int argc, char** argv) {

#define N atoi(argv[1])

#define debug 1
	
	size_t bytes = sizeof(myFloat) * N * N;
	dim3 BLOCK_SIZE(N, N, 1);

	std::chrono::duration<double, std::milli> millisecondsCPUhost;

	cudaEvent_t startGPU;
	cudaEvent_t stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	float milliseconds;

	Matrix* h_A, * h_B, * h_C, * hostRes_C;
	myFloat* d_A, * d_B, * d_C;

	h_A = new Matrix(N);
	h_B = new Matrix(N);
	h_C = new Matrix(N);
	hostRes_C = new Matrix(N);

	cudaMalloc((void**)&d_A, bytes);
	cudaMalloc((void**)&d_B, bytes);
	cudaMalloc((void**)&d_C, bytes);

	*h_A = Matrix::idMatrix(N);
	*h_B = Matrix::idMatrix(N);
	*h_C = Matrix::nullMatrix(N);
	*hostRes_C = Matrix::nullMatrix(N);

	cudaMemcpy((void*)d_A, (void*)h_A->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_B, (void*)h_B->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_C, (void*)h_C->content, bytes, cudaMemcpyHostToDevice);

	
	auto startCPUhost = std::chrono::high_resolution_clock::now();
	*hostRes_C = *h_A + *h_B;
	auto stopCPUhost = std::chrono::high_resolution_clock::now();

	millisecondsCPUhost = stopCPUhost - startCPUhost;

	cudaEventRecord(startGPU);
	matrixAddV1 <<<1, BLOCK_SIZE >>> (d_A, d_B, d_C, N);
	cudaEventRecord(stopGPU);

	cudaEventSynchronize(stopGPU);

	cudaMemcpy(h_C->content, (void*)d_C, bytes, cudaMemcpyDeviceToHost);
	

	cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);

	if (debug) {
		std::cout << "CPU result : " << std::endl << std::endl;
		hostRes_C->display();
		std::cout << "GPU result : " << std::endl << std::endl;
		h_C->display();
	}
	

	std::cout << std::endl << "Matrix addition of " << N << " elements took " << millisecondsCPUhost.count() << " ms to complete on the CPU. " << std::endl << std::endl;
	std::cout << std::endl << "Matrix addition of " << N << " elements took " << milliseconds << " ms to complete on the GPU. " << std::endl << std::endl;

	return 0;
}