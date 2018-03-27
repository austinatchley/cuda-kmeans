CFLAGS:=
export CFLAGS
NVCC = nvcc
NVCCFLAGS = $(CFLAGS) -I/opt/cuda-8.0/include -L/opt/cuda-8.0/lib64 -lcuda -lcudart --ptxas-options=-v -std=c++11

%.o : %.c++
		g++ $(CFLAGS) -c $< -std=c++11

CUDA_CPP_SRC = kmeans-main.c++

CUDA_CPP_OBJ = $(CUDA_CPP_SRC:%.c++=%.o)

all: kmeans-1

kmeans.out: $(CUDA_CPP_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(CFLAGS) -o $@ $(CUDA_CPP_OBJ) $(CUDA_CU_OBJ)

kmeans-1: $(CUDA_CPP_OBJ) cuda.o
	 g++ -L/opt/cuda-8.0/lib64 $(CUDA_CPP_OBJ) kmeans-cuda.o -o kmeans.out -std=c++11 -lcuda -lcudart

cuda.o:
	nvcc $(CFLAGS) -c -L/opt/cuda-8.0/lib64 kmeans-cuda.cu -std=c++11 -arch=sm_61




format:
	clang-format -i *.c++ *.cu
	clang-format -i *.h 

add: format yapf
	git add .

yapf:
	yapf -i timer_harness.py harness.py 

clean:
	rm -rf *.out *.o

