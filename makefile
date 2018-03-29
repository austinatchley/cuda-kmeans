CFLAGS=

NVCC = nvcc
NVCCFLAGS = $(CFLAGS) -I/opt/cuda-8.0/include -L/opt/cuda-8.0/lib64 -lcuda -lcudart --ptxas-options=-v -std=c++11

%.o : %.c++
		g++ $(CFLAGS) -c $< -std=c++11

CUDA_CPP_SRC = kmeans-main.c++

CUDA_CPP_OBJ = $(CUDA_CPP_SRC:%.c++=%.o)

all: kmeans

kmeans: $(CUDA_CPP_OBJ)
	$(NVCC) $(CFLAGS) -arch=sm_61 -std=c++11 -g -O0 kmeans-main.o kmeans-cuda.cu -o kmeans.out




format:
	clang-format -i *.c++ *.cu
	clang-format -i *.h 

add: format yapf
	git add .

yapf:
	yapf -i timer_harness.py harness.py 

clean:
	rm -rf *.out *.o

