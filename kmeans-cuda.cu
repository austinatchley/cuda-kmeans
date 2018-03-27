#include "kmeans.h"

#include <cuda_runtime_api.h>
#ifdef PROFILE
#include <cuda_profiler_api.h>
#endif

__global__ static void find_nearest_cluster(double **points, double **centroids,
                                            int num_points, int num_coords,
                                            int num_centroids) {
  extern __shared__ char shared[]; // array of bytes of shared memory

  int point_id = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%d\n", point_id);
}

__global__ static void compute_change(double **centroids,
                                      double **old_centroids, int num_centroids,
                                      int num_coords) {
  extern __shared__ char shared[]; // array of bytes of shared memory

  int point_id = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%d\n", point_id);
}

void kmeans(double **points, double **centroids, double **old_centroids,
            int num_points, int num_coords, int num_centroids,
            int max_iterations, double threshold) {

  double *dev_points;
  double *dev_centroids;

  const size_t threads_per_block = 128; // This is a design decision
  const size_t num_blocks =
      (num_points + threads_per_block - 1) / threads_per_block;
  const size_t shared_mem_per_block = threads_per_block * sizeof(char);

  cout << "kmeans" << endl;

  cudaMalloc((void **)&dev_points, num_points * num_coords * sizeof(double));
  cudaMalloc((void **)&dev_centroids,
             num_centroids * num_coords * sizeof(double));

  cudaMemcpy(dev_points, points[0], num_points * num_coords * sizeof(double),
             cudaMemcpyHostToDevice);

  do {
    cudaMemcpy(dev_centroids, centroids[0],
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);

    find_nearest_cluster<<<num_blocks, threads_per_block,
                           shared_mem_per_block>>>(
        points, centroids, num_points, num_coords, num_centroids);

    cudaDeviceSynchronize();

    // compute_change
    //  <<<1, reduction_threads, shared_mem_for_reduction>>>
    //  (centroids, old_centroids, num_centroids, num_coords);
  } while (true);
}
