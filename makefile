all: kmeans 


kmeans: cuda.o
	 g++ -o kmeans.out -L/usr/local/cuda/lib64 -lcuda kmeans-io.c++ kmeans-main.c++  kmeans-cuda.o -std=c++11

cuda.o:
	nvcc -c -arch=sm_20 kmeans-cuda.cu -std=c++11




format:
	clang-format -i kmeans.cu 
	clang-format -i kmeans.h 

add: format yapf
	git add .

yapf:
	yapf -i timer_harness.py harness.py 

clean:
	rm -rf *.out *.o

