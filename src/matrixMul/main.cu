#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"
#include <iostream>

bool check(myFloat* A, myFloat* B, unsigned int N, double epsilon) {
	double errMax = 0.0;
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			errMax = max(abs(A[i * N + j] - B[i * N + j]), errMax);
			if(errMax > epsilon){
				return false;
			}
		}
	}
	return true;
}

void display(myFloat* A, unsigned int N){
	for(size_t i = 0; i < N; i++){
		for(size_t j = 0; j < N; j++){
			std::cout << A[i * N + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

int main(int argc, char** argv) {


#define N atoi(argv[1])
#define debug 1
#define ITER 10
	
	size_t bytes = sizeof(myFloat) * N * N;
	dim3 BLOCK_SIZE(THREADS_NUMBER, THREADS_NUMBER, 1);
	dim3 GRID_SIZE((N/THREADS_NUMBER), (N / THREADS_NUMBER), 1);

	cudaEvent_t startGPU, startGPUtiled;
	cudaEvent_t stopGPU, stopGPUtiled;
	cudaEventCreate(&startGPUtiled); cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPUtiled); cudaEventCreate(&stopGPU);
	float millisecondsV3, millisecondsV4;
	std::chrono::duration<double, std::milli> millisecondsCPUinit, millisecondsDeviceHostCopy;

	std::cout << "Matrix multiplication of " << N / 1000 << "K elements, using " << THREADS_NUMBER << " threads per block. \n\n";
	std::cout << "Iteration " << " | " << "host matrix init time" << " | " << "GPU tiled exec time" << " | " << "GPU V4 exec time" << " | " << "device -> host copy duration \n\n";
	
	for (size_t i = 0; i < ITER; i++)
	{
		Matrix* h_A, * h_B, * h_C, * h_C_tiled;
		myFloat* d_A, * d_B, * d_C, * d_C_tiled;

		h_A = new Matrix(N);
		h_B = new Matrix(N);
		h_C = new Matrix(N);
		h_C_tiled = new Matrix(N);

		cudaMalloc((void**)&d_A, bytes);
		cudaMalloc((void**)&d_B, bytes);
		cudaMalloc((void**)&d_C, bytes);
		cudaMalloc((void**)&d_C_tiled, bytes);

		auto startCPU = std::chrono::high_resolution_clock::now();
		h_A->randMatrix(0, 1);
		h_B->randMatrix(0, 1);
		h_C->nullMatrix();
		h_C_tiled->nullMatrix();
		auto stopCPU = std::chrono::high_resolution_clock::now();

		millisecondsCPUinit = stopCPU - startCPU;

		cudaMemcpy((void*)d_A, (void*)h_A->content, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_B, (void*)h_B->content, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_C, (void*)h_C->content, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_C_tiled, (void*)h_C_tiled->content, bytes, cudaMemcpyHostToDevice);

		//Tiled matrix multiplication
		cudaEventRecord(startGPUtiled);
		matrixMulV2<< <GRID_SIZE, BLOCK_SIZE >> > (d_A, d_B, d_C_tiled, N);
		cudaEventRecord(stopGPUtiled);

		cudaEventSynchronize(stopGPUtiled);

		auto startDeviceHostCopy = std::chrono::high_resolution_clock::now();
		cudaMemcpy((void*)h_C_tiled->content, (void*)d_C_tiled, bytes, cudaMemcpyDeviceToHost);
		auto stopDeviceHostCopy = std::chrono::high_resolution_clock::now();

		millisecondsDeviceHostCopy = stopDeviceHostCopy - startDeviceHostCopy;

		//GPU Matrix multiplication with prefetch
		cudaEventRecord(startGPU);
		matrixMulV3 << <GRID_SIZE, BLOCK_SIZE >> > (d_A, d_B, d_C, N);
		cudaEventRecord(stopGPU);

		cudaEventSynchronize(stopGPU);

		cudaMemcpy((void*)h_C->content, (void*)d_C, bytes, cudaMemcpyDeviceToHost);

		cudaEventElapsedTime(&millisecondsV3, startGPUtiled, stopGPUtiled);
		cudaEventElapsedTime(&millisecondsV4, startGPU, stopGPU);

		if (debug) {

			if (check(h_C->content, h_C_tiled->content, N, 0.001)) {
				std::cout << "The operation is correct. \n\n";
			}
			else {
				std::cout << "The operation is incorrect. \n\n";
			}
		}

		std::cout << "     " << i << "    " << " |       " << millisecondsCPUinit.count() << " ms " << "     |    " << millisecondsV3 << " ms " << "   |    " << millisecondsV4 << " ms " << "  |    " << millisecondsDeviceHostCopy.count() << " ms \n\n";

		//Freeing the memory

		cudaFree((void*)d_A); cudaFree((void*)d_B); cudaFree((void*)d_C); cudaFree((void*)d_C_tiled);
		free(h_A); free(h_B); free(h_C); free(h_C_tiled);
		d_A = nullptr; d_B = nullptr; d_C = nullptr; d_C_tiled = nullptr;
		h_A = nullptr; h_B = nullptr; h_C = nullptr; h_C_tiled = nullptr;

	}
		
	return 0;
} 
