CFLAGS:=
export CFLAGS
NVCC = nvcc
NVCCFLAGS = $(CFLAGS) -I/opt/cuda-8.0/include -L/opt/cuda-8.0/lib64 -lcuda -lcudart --ptxas-options=-v -std=c++11

%.o : %.cu
		$(NVCC) $(NVCCFLAGS) -o $@ -c $<

CUDA_CPP_SRC = kmeans-main.cu
CUDA_CU_SRC = kmeans-cuda.cu

CUDA_CPP_OBJ = $(CUDA_CPP_SRC:%.cu=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)

all: kmeans.out

kmeans.out: $(CUDA_CPP_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(CFLAGS) -o $@ $(CUDA_CPP_OBJ) $(CUDA_CU_OBJ)

kmeans-1: cuda.o
	 g++ -c -I/opt/cuda-8.0/include *.c++ -std=c++11
	 nvcc -o kmeans.out -L/opt/cuda-8.0/lib64 -lcuda -lcudart *.o 

cuda.o:
	nvcc $(CFLAGS) -c -I/opt/cuda-8.0/include -L/opt/cuda-8.0/lib64 kmeans-cuda.cu -std=c++11




format:
	clang-format -i *.c++ *.cu
	clang-format -i *.h 

add: format yapf
	git add .

yapf:
	yapf -i timer_harness.py harness.py 

clean:
	rm -rf *.out *.o

