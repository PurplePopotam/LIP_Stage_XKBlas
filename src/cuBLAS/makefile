CC=nvcc
NVCCFLAGS=-lcublas
EXEC=main

all: $(EXEC)

main: main.o
	$(CC) main.c $(NVCCFLAGS) -o main

clean: 
	rm -rf *.o

veryclean: clean
	rm -rf $(EXEC)
