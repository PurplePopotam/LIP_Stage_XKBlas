#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

typedef float myFloat;

class Matrix {

	public:
		Matrix(const size_t& _width);
		Matrix(const Matrix& other);
		Matrix(Matrix&& other);
		inline ~Matrix() { delete content; }
		inline myFloat at(const size_t& i, const size_t& j) const { return content[i * width + j]; }
		inline myFloat& at(const size_t& i, const size_t& j) { return content[i * width + j]; }

		Matrix operator=(const Matrix& other);
		Matrix& operator=(Matrix&& other);
		Matrix operator+(const Matrix& other);

		Matrix operator*(const Matrix& other);
		myFloat* operator*(myFloat* other);

		void display(unsigned int n);

		void nullMatrix();
		void idMatrix();
		void randMatrix(const myFloat& min, const myFloat& max);
		void sparseMatrix(const float& r); //r represents the matrix occupation rate

		inline size_t getWidth() { return width; }

	public:
		myFloat* content;
		size_t width;
};



#endif
