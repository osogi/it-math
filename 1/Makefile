CC = gcc

.PHONY: clean all

all: main
	
main: main.c Makefile
	gcc main.c -fopenmp -lm -o main.elf

clean:
	rm -rf main.elf
