#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"
#include <iostream>


bool check(myFloat* v, myFloat* w, unsigned int N, double epsilon) {
	double errMax = 0.0;
	for (size_t i = 0; i < N; i++)
	{
		errMax = max(abs(v[i] - w[i]), errMax);
		if (errMax > epsilon) {
			return false;
		}
	}
	return true;
}

void initVec(myFloat* v, unsigned int N, myFloat min, myFloat max) {
	for (size_t i = 0; i < N; i++)
	{
		v[i] = rand() % int(max) + myFloat(rand()) / RAND_MAX + min;
	}
}

void display(myFloat* v, unsigned int N) {

	for (size_t i = 0; i < N; i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << "\n";
}



int main(int argc, char** argv) {

#define N atoi(argv[1])
#define debug 0

	size_t bytesMatrix = sizeof(myFloat) * N * N;
	size_t bytesVector = sizeof(myFloat) * N;
	dim3 BLOCK_SIZE(THREADS_NUMBER, THREADS_NUMBER, 1);
	dim3 GRID_SIZE((N / THREADS_NUMBER), (N / THREADS_NUMBER), 1);

	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU); cudaEventCreate(&stopGPU);
	float milliseconds;
	std::chrono::duration<double, std::milli> millisecondsCPU;

	myFloat* v, res;
	myFloat* d_A, d_v, d_res;

	cudaMalloc((void**)&d_A, bytesMatrix);
	cudaMalloc((void**)&d_v, bytesVector);
	cudaMalloc((void**)&d_res, bytesVector);

	A = new Matrix(N);
	v = new myFloat[N];
	res = new myFloat[N];
	A.randMatrix(0, 1);
	initVec(v, N, 0, 1);


	cudaMemcpy((void*)d_A, (void*)A->content, bytesMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_v, (void*)v, bytesVector, cudaMemcpyHostToDevice);
	
	res = A * v;
	A.display(N);
	display(v, N);
	display(res, N);

	return 0;
}