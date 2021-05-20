CC=NVCC
NVCCFLAGS=
EXEC=main

all: $(EXEC)

main: matrix.o kernels.o main.o
	$(CC) -o main main.obj matrix.obj kernels.obj

kernels.o: kernels.cu
	$(CC) -c kernels.cu $(NVCCFLAGS)

matrix.o: matrix.cu
	$(CC) -c matrix.cu $(NVCCFLAGS)

main.o: main.cu kernels.cuh matrix.cuh
	$(CC) -c main.cu $(NVCCFLAGS)

clean: 
	rm -rf *.obj

veryclean: clean
	rm -rf $(EXEC)
