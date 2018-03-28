#include "kmeans.h"

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void cudaCheckError(const char *msg);

__global__ static void find_nearest_cluster(double **points, double **centroids, int *cluster,
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
            int num_points, int num_coords, int num_centroids, int *cluster,
            int *cluster_size, int max_iterations, double threshold) {

  double *dev_points;
  double *dev_centroids;
  int *dev_cluster;

  int iterations = 0;

  for (int i = 0; i < num_points; ++i)
    cluster[i] = -1; // init cluster membership to default val

  const size_t threads_per_block = 128; // This is simply a design decision
  const size_t num_blocks =
      (num_points + threads_per_block - 1) / threads_per_block;
  const size_t shared_mem_per_block = threads_per_block * sizeof(char);

  cout << "kmeans" << endl;

  // cudaSetDevice(0);

  cudaMalloc((void **)&dev_points, num_points * num_coords * sizeof(double));
  cudaCheckError("malloc dev_points");

  cudaMalloc((void **)&dev_centroids,
             num_centroids * num_coords * sizeof(double));
  cudaCheckError("malloc dev_centroids");

  cudaMalloc((void **)&dev_cluster, num_points * sizeof(int));
  cudaCheckError("malloc dev_cluster");

  cudaMemcpy(dev_points, points[0], num_points * num_coords * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaCheckError("copy points to device");

  cudaMemcpy(dev_cluster, cluster, num_points * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaCheckError("copy cluster to device");

  do {
    cudaMemcpy(dev_centroids, centroids[0],
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);

    find_nearest_cluster<<<num_blocks, threads_per_block,
                           shared_mem_per_block>>>(
        dev_points, dev_centroids, dev_cluster, num_points, num_coords, num_centroids);

    cudaDeviceSynchronize();

    cudaMemcpy(cluster, dev_cluster, num_points * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaCheckError("copy clusters back to host");

    for (int i = 0; i < num_points; ++i) {
      int cluster_idx = cluster[i];
      assert(cluster_idx >= 0);

      // increment cluster_size
      ++cluster_size[cluster_idx];
      for (int j = 0; j < num_coords; ++j)
        centroids[cluster_idx][j] += points[i][j];
    }

    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        if (cluster_size[i] > 0)
          centroids[i][j] /= cluster_size[i];
      }
    }

    // compute_change
    //  <<<1, reduction_threads, shared_mem_for_reduction>>>
    //  (centroids, old_centroids, num_centroids, num_coords);
    // cudaDeviceSynchronize();

    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; i < num_coords; ++j) {
        old_centroids[i][j] = centroids[i][j];
      }
    }

    ++iterations;
  } while (true);
}

void cudaCheckError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
