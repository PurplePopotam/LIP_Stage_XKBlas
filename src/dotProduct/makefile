CC=NVCC
NVCCFLAGS=
EXEC=main

all: $(EXEC)

main: kernels.o main.o
	$(CC) -o main main.o kernels.o

kernels.o: kernels.cu
	$(CC) -c kernels.cu $(NVCCFLAGS)

main.o: main.cu kernels.cuh
	$(CC) -c main.cu $(NVCCFLAGS)

clean: 
	rm -rf *.o

veryclean: clean
	rm -rf $(EXEC)
