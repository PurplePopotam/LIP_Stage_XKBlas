CC=nvcc
NVCCFLAGS=
EXEC=main
BLAS_LIBDIR=${MKLROOT}/lib
BLAS_LIB_SO=${MKLROOT}/lib/intel64/libmkl_intel_ilp64.so
BLAS_CPPFLAGS=-DMKL_ILP64 -m64 -I${MKLROOT}/include
BLAS_LDFLAGS=-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

all: $(EXEC)

main: matrix.o kernels.o main.o
	$(CC) -o main main.o matrix.o kernels.o ${BLAS_LDFLAGS}

kernels.o: kernels.cu
	$(CC) -c kernels.cu ${NVCCFLAGS} ${BLAS_CPPFLAGS} 

matrix.o: matrix.cu
	$(CC) -c matrix.cu ${NVCCFLAGS} ${BLAS_CPPFLAGS}

main.o: main.cu kernels.cuh matrix.cuh
	$(CC) -c main.cu ${NVCCFLAGS} ${BLAS_CPPFLAGS}

clean: 
	rm -rf *.obj && rm -rf *.o

veryclean: clean
	rm -rf $(EXEC)
