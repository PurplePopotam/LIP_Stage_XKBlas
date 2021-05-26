#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"
#include <iostream>
#include "mkl.h"

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

	std::chrono::duration<double, std::milli> millisecondsCPUhost;

	cudaEvent_t startGPU;
	cudaEvent_t stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	float milliseconds;
	
	double alpha, beta;

	Matrix* h_A, * h_B, * h_C;
	myFloat* d_A, * d_B, * d_C;
	myFloat* cpu_A, *  cpu_B, *  cpu_C;

	h_A = new Matrix(N);
	h_B = new Matrix(N);
	h_C = new Matrix(N);

	cudaMalloc((void**)&d_A, bytes);
	cudaMalloc((void**)&d_B, bytes);
	cudaMalloc((void**)&d_C, bytes);

	cpu_A = (myFloat*)mkl_malloc(bytes, 64);
	cpu_B = (myFloat*)mkl_malloc(bytes, 64);
	cpu_C = (myFloat*)mkl_malloc(bytes, 64);
	
	if (cpu_A == NULL || cpu_B == NULL || cpu_C == NULL) {
		std::cout << "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n";
		mkl_free(cpu_A);
		mkl_free(cpu_B);
		mkl_free(cpu_C);
		return 1;
	}	
	
	alpha = 1.0;
	beta = 0.0;

	std::cout << "Initializing Matrix data...\n\n";

	*h_A = Matrix::randMatrix(N);
	*h_B = Matrix::randMatrix(N);
	*h_C = Matrix::nullMatrix(N);

	for(size_t i = 0; i < N*N; i++){
		cpu_A[i] = rand()%10;
		cpu_B[i] = rand()%10;	
		cpu_C[i] = rand()%10;
	}

	cudaMemcpy((void*)d_A, (void*)h_A->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_B, (void*)h_B->content, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_C, (void*)h_C->content, bytes, cudaMemcpyHostToDevice);

	//CPU Matrix Multiplication
	auto startCPUhost = std::chrono::high_resolution_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, cpu_A, N, cpu_B, N, beta, cpu_C, N);
	auto stopCPUhost = std::chrono::high_resolution_clock::now();

	millisecondsCPUhost = stopCPUhost - startCPUhost;
	
	//GPU Matrix Multiplication
	cudaEventRecord(startGPU);
	matrixMulV3<<<GRID_SIZE, BLOCK_SIZE>>> (d_A, d_B, d_C, N);
	cudaEventRecord(stopGPU);
	

	cudaEventSynchronize(stopGPU);

	cudaMemcpy(h_C->content, (void*)d_C, bytes, cudaMemcpyDeviceToHost);

	cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);

	if (debug) {
		std::cout << "CPU result : " << std::endl << std::endl;
		display(cpu_C, N);
		std::cout << "GPU result : " << std::endl << std::endl;
		h_C->display();
	}
	
	//check(h_C->content,cpu_C, N);
	
	mkl_free(cpu_A); mkl_free(cpu_B); mkl_free(cpu_C);
	cudaFree((void*)d_A); cudaFree((void*)d_B); cudaFree((void*)d_C);
	d_A = nullptr; d_B = nullptr; d_C = nullptr;

	std::cout << std::endl << "Matrix multiplication of " << N << " elements took " << millisecondsCPUhost.count() << " ms to complete on the CPU. " << std::endl << std::endl;
	std::cout << std::endl << "Matrix multiplication of " << N << " elements took " << milliseconds << " ms to complete on the GPU. " << std::endl << std::endl;
	//Test push multiple remotes
	return 0;
} 
