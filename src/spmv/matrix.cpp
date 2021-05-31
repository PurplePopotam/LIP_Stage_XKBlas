#include <stdio.h>
#include "matrix.hpp"
#include <iostream>

Matrix::Matrix(const size_t& _width) {
	width = _width;
	content = new myFloat[width * width];
}

Matrix::Matrix(const Matrix& other) {
	width = other.width;
	content = new myFloat[width * width];
	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			at(i, j) = other.at(i, j);
		}
	}
}

Matrix::Matrix(Matrix&& other) {
	content = other.content;
	width = other.width;
	other.width = 0;
	other.content = nullptr;
}

void Matrix::display(unsigned int n) {
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			std::cout << at(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

Matrix Matrix::operator=(const Matrix& other) {
	if (content == other.content) {
		return *this;
	}
	else {
		for (size_t i = 0; i < width; i++)
		{
			for (size_t j = 0; j < width; j++)
			{
				at(i, j) = other.at(i, j);
			}
		}
	}
	return *this;
}

Matrix& Matrix::operator=(Matrix&& other) {
	if (this != &other) {
		delete content;
		content = other.content;
		width = other.width;
		other.content = nullptr;
		other.width = 0;
	}

	return *this;
}

Matrix Matrix::operator+(const Matrix& other) {
	Matrix res(width);

	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			res.at(i, j) = at(i,j) + other.at(i, j);
		}
	}
	return res;
}

Matrix Matrix::operator*(const Matrix& other) {
	Matrix res(width);

	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			for (size_t k = 0; k < width; k++)
			{
				res.at(i, j) += at(i, k)*other.at(k, j);
			}
		}
	}
	return res;
}

Matrix Matrix::operator*(const myFloat& other) {
	myFloat* res = new myFloat[this.width];

	for (size_t i = 0; i < width; i++)
	{
		for (size_t k = 0; k < width; k++)
		{
			res[i] += at(i, k) * other[k];
		}
	}
	return res;
}

void Matrix::idMatrix() {
	//#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			if (i == j) {
				this->at(i, j) = 1;
			}
			else {
				this->at(i, j) = 0;
			}
		}
	}
}

void Matrix::nullMatrix() {
	//#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			this->at(i, j) = 0;
		}
	}
}

void Matrix::randMatrix(const myFloat& min, const myFloat& max) {
	//#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			this->at(i, j) = rand()%int(max) + myFloat(rand()) / RAND_MAX + min;
		}
	}
}

void Matrix::sparseMatrix(const float& r) {
	//#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			if (float(rand() % 100) / 100.0 < r) {
				this->at(i, j) = rand() % 10 + myFloat(rand()) / RAND_MAX;
			}
			else {
				this->at(i, j) = 0;
			}
		}
	}
}
