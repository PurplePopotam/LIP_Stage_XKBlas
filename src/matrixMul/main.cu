#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"
#include <iostream>


int main(int argc, char** argv) {

	#define N atoi(argv[1])

	Matrix A(N), B(N), C(N);
	std::chrono::duration<double, std::milli> millisecondsCPUhost;


	A = Matrix::idMatrix(N);
	B = Matrix::idMatrix(N);

	auto startCPUhost = std::chrono::high_resolution_clock::now();
	C = A + B;
	auto stopCPUhost = std::chrono::high_resolution_clock::now();

	millisecondsCPUhost = stopCPUhost - startCPUhost;

	std::cout << std::endl << "Matrix addition of " << N << " elements took " << millisecondsCPUhost.count() << " ms to complete on the CPU. " << std::endl;

	return 0;
}