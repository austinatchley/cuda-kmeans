CFLAGS:=
export CFLAGS

all: kmeans 


kmeans: cuda.o
	 g++ -c -I/opt/cuda-8.0/include *.c++ -std=c++11
	 nvcc -o kmeans.out -L/opt/cuda-8.0/lib64 -lcuda -lcudart *.o 

cuda.o:
	nvcc $(CFLAGS) -c -I/opt/cuda-8.0/include -L/opt/cuda-8.0/lib64 kmeans-cuda.cu -std=c++11




format:
	clang-format -i *.c *.cu
	clang-format -i *.h 

add: format yapf
	git add .

yapf:
	yapf -i timer_harness.py harness.py 

clean:
	rm -rf *.out *.o

