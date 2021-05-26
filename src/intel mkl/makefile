CC=g++
NVCCFLAGS=
EXEC=main
BLAS_LIBDIR=${MKLROOT}/lib
BLAS_LIB_SO=${MKLROOT}/lib/intel64/libmkl_intel_ilp64.so
BLAS_CPPFLAGS=-DMKL_ILP64 -m64 -I${MKLROOT}/include
BLAS_LDFLAGS=-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

all: $(EXEC)

main: main.o
	$(CC) -o main main.o ${BLAS_LDFLAGS}


main.o: main.c
	$(CC) -c main.c ${NVCCFLAGS} ${BLAS_CPPFLAGS}

clean: 
	rm -rf *.obj && rm -rf *.o

veryclean: clean
	rm -rf $(EXEC)
