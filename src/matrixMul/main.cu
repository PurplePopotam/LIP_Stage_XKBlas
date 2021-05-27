#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"
#include <iostream>

void check(myFloat* A, myFloat* B, unsigned int N) {
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			assert(A[i * N + j] == B[i * N + j]);
		}
	}
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
#define debug 0
	
	size_t bytes = sizeof(myFloat) * N * N;
	dim3 BLOCK_SIZE(THREADS_NUMBER, THREADS_NUMBER, 1);
	dim3 GRID_SIZE((N/THREADS_NUMBER) + 1, (N / THREADS_NUMBER) + 1, 1);

	cudaEvent_t startGPU, startGPUtiled;
	cudaEvent_t stopGPU, stopGPUtiled;
	cudaEventCreate(&startGPUtiled); cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPUtiled); cudaEventCreate(&stopGPU);
	float millisecondsTiled, milliseconds;

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

	std::cout << "Initializing Matrix data...\n\n";

	*h_A = Matrix::randMatrix(N);
	*h_B = Matrix::randMatrix(N);
	*h_C = Matrix::nullMatrix(N);

	std::cout << "Done initialazing. \n\n";

	cudaMemcpy((void*)d_A, (void*)h_A->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_B, (void*)h_B->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_C, (void*)h_C->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_C_tiled, (void*)h_C->content, bytes, cudaMemcpyHostToDevice);

	//GPU tiled Matrix Multiplication
	cudaEventRecord(startGPUtiled);
	matrixMulV2<<<GRID_SIZE, BLOCK_SIZE>>> (d_A, d_B, d_C_tiled, N);
	cudaEventRecord(stopGPUtiled);
	
	cudaEventSynchronize(stopGPUtiled);

	//GPU regular Matrix Multiplication with small optimizations
	cudaEventRecord(startGPU);
	matrixMulV4i<<<GRID_SIZE, BLOCK_SIZE>>> (d_A, d_B, d_C, N);
	cudaEventRecord(stopGPU);

	cudaMemcpy(h_C_tiled->content, (void*)d_C_tiled, bytes, cudaMemcpyDeviceToHost);

	cudaEventElapsedTime(&millisecondsTiled, startGPUtiled, stopGPUtiled);

	if (debug) {
		std::cout << "GPU result : " << std::endl << std::endl;
		h_C->display();
		std::cout << "GPU tiled result : " << std::endl << std::endl;
		h_C_tiled->display();
	}
	
	check(h_C->content,h_C_tiled->content, N);
	
	cudaFree((void*)d_A); cudaFree((void*)d_B); cudaFree((void*)d_C);
	d_A = nullptr; d_B = nullptr; d_C = nullptr;

	std::cout << std::endl << "Matrix multiplication of " << N << " elements took " << millisecondsTiled << " ms to complete on the GPU. " << std::endl << std::endl;

	return 0;
} 
