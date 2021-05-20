CC=NVCC
NVCCFLAGS=
EXEC=main

all: $(EXEC)

main: kernels.o main.o
	$(CC) -o main main.obj kernels.obj

kernels.o: kernels.cu
	$(CC) -c kernels.cu $(NVCCFLAGS)

main.o: main.cu kernels.cuh
	$(CC) -c main.cu $(NVCCFLAGS)

clean: 
	rm -rf *.obj

veryclean: clean
	rm -rf $(EXEC)
