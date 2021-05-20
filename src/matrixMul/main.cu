#include <cassert>
#include <device_launch_parameters.h>
#include <chrono>
#include "kernels.cuh"



int main(int argc, char** argv) {

	#define N atoi(argv[1])

	Matrix A(N);
	A = Matrix::idMatrix(N);
	A.display();

	return 0;
}