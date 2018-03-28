#include "kmeans.h"

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void cudaCheckError(const char *msg);

__device__ inline static double
euclidian_dist_squared(double *points, double *centroids, int num_points,
                       int num_centroids, int num_coords, int point_id,
                       int centroid_id) {
  assert(point_id < num_points);
  assert(centroid_id < num_centroids);

  int i = 0;
  double dist = 0.0;

  for (; i < num_coords; ++i)
    dist += powf(points[point_id * num_coords + i] -
                     centroids[centroid_id * num_coords + i],
                 2);

  return dist;
}

__global__ static void find_nearest_cluster(double *dev_points,
                                            double *dev_centroids,
                                            int *dev_cluster, int num_points,
                                            int num_coords, int num_centroids) {
  //extern __shared__ char shared[]; // array of bytes of shared memory

  int point_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (point_id == 0) {
    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        printf("%d ", dev_centroids[i*num_coords + j]);
      }
      printf("\n");
    }
    printf("\n");
  }



  if (point_id >= num_points)
    return;

  // start with the dist between the point and the first centroid
  double min_dist =
      euclidian_dist_squared(dev_points, dev_centroids, num_points, num_centroids,
                             num_coords, point_id, 0);
  int min_index = 0;

  double dist;

  // start at 1 because we already calculated the 0th index
  for (int i = 1; i < num_centroids; ++i) {
    dist = euclidian_dist_squared(dev_points, dev_centroids, num_points,
                                  num_centroids, num_coords, point_id, i);

    if (dist < min_dist) {
      min_index = i;
      min_dist = dist;
    }
  }

  if(min_index == 11)
    printf("point %d closest to c11 (%f,%f,%f) with coord (%f,%f,%f)\n",
        point_id, dev_centroids[11*num_coords], dev_centroids[11*num_coords + 1], dev_centroids[11*num_coords + 2],
        dev_points[point_id*num_coords], dev_points[point_id*num_coords + 1],  dev_points[point_id*num_coords + 2]);
  else
    printf("point %d closest to c%d (%f,%f,%f) with coord (%f,%f,%f)\n",
        point_id, min_index, dev_centroids[min_index*num_coords], dev_centroids[min_index*num_coords + 1], dev_centroids[min_index*num_coords + 2],
        dev_points[point_id*num_coords], dev_points[point_id*num_coords + 1],  dev_points[point_id*num_coords + 2]);

  dev_cluster[point_id] = min_index;
}

__global__ static void compute_change(double **centroids,
                                      double **old_centroids, int num_centroids,
                                      int num_coords) {
  //extern __shared__ char shared[]; // array of bytes of shared memory

  int point_id = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%d\n", point_id);
}

double **kmeans(double ** const points, double **centroids, double **old_centroids,
            int num_points, int num_coords, int num_centroids, int * const cluster,
            int *cluster_size, int *num_iterations, int max_iterations, double threshold) {

  double *dev_points;
  double *dev_centroids;
  int *dev_cluster;

  int iterations = 0;

  for (int i = 0; i < num_points; ++i)
    cluster[i] = -1; // init cluster membership to default val

  for (int i = 0; i < num_centroids; ++i) 
    cluster_size[i] = 0;

  const size_t threads_per_block = 128; // This is simply a design decision
  const size_t num_blocks =
      (num_points + threads_per_block - 1) / threads_per_block;
  const size_t shared_mem_per_block = threads_per_block * sizeof(char);

  cudaSetDevice(0);

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
    // prints every iteration of centroids
    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        cout << centroids[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;

   cudaMemcpy(dev_centroids, centroids[0],
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);

    find_nearest_cluster<<<num_blocks, threads_per_block,
                           shared_mem_per_block>>>(dev_points, dev_centroids,
                                                   dev_cluster, num_points,
                                                   num_coords, num_centroids);

    cudaDeviceSynchronize();

    cudaMemcpy(cluster, dev_cluster, num_points * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaCheckError("copy point->cluster map back to host");

    for (int i = 0; i < num_centroids; ++i)
      for (int j = 0; j < num_coords; ++j)
        centroids[i][j] = 0;

    for (int i = 0; i < num_points; ++i) {
      int cluster_idx = cluster[i];
      assert(cluster_idx >= 0 && cluster_idx < num_centroids);

      // increment cluster_size
      ++cluster_size[cluster_idx];
      assert(cluster_size[cluster_idx] < num_points);
      for (int j = 0; j < num_coords; ++j)
        centroids[cluster_idx][j] += points[i][j];
    }

    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        if (cluster_size[i] > 0) {
          centroids[i][j] /= cluster_size[i];
        } else {
          assert(0);
          //centroids[i][j] = old_centroids[i][j];
        }
      }
      cluster_size[i] = 0;
    }

    // compute_change
    //  <<<1, reduction_threads, shared_mem_for_reduction>>>
    //  (centroids, old_centroids, num_centroids, num_coords);
    // cudaDeviceSynchronize();

    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        old_centroids[i][j] = centroids[i][j];
      }
    }

    ++iterations;
  } while (iterations < max_iterations);
  
  cudaFree(dev_points);
  cudaFree(dev_centroids);
  cudaFree(dev_cluster);

  *num_iterations = iterations;
  return centroids;
}

void cudaCheckError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
