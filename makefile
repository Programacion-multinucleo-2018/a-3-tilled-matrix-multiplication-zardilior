main:
	nvcc -std=c++11 original.cu -Xcompiler -fopenmp -lgomp\
	   	-o bin/program

clean:
	rm test/bin/* bin/*
