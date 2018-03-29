#include "kmeans.h"

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/*
 * Global
 */

__device__ bool converged = false;

/*
 * Prototypes
 */

void cudaCheckError(const char *msg);
void flatten_2D_array(double **src, double *dest, int r, int c);
void unflatten_1D_array(double *src, double **dest, int r, int c);

/*
 * CUDA Kernels
 */

__device__ inline static double
euclidian_dist_squared(double *points, double *centroids, int num_points,
                       int num_centroids, int num_coords, int point_id,
                       int centroid_id) {
  assert(point_id < num_points);
  assert(centroid_id < num_centroids);

  double dist = 0.0;

  for (int i = 0; i < num_coords; ++i) {
    double x = points[point_id * num_coords + i] - centroids[centroid_id * num_coords + i];
    dist += x * x;
  }

  return dist;
}

__device__ static void accum_centroids(double *dev_points, double *dev_centroids, int *dev_cluster, int num_points, int num_coords, int num_centroids) {
  
  // shared with thread block
  extern __shared__ double shared_centroids[];

  int point_id = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double *centroids = dev_centroids;

#ifdef SHARED_MEM
  if (point_id == 0)
    for (int i = 0; i < num_centroids; ++i)
      for (int j = 0; j < num_coords; ++j)
        shared_centroids[i*num_coords + j] = 0.0;

  __syncthreads();
  centroids = shared_centroids;
#endif

  for (int i = point_id; i < num_points; i+=stride) {
    int centr_idx = dev_cluster[i];
    for (int j = 0; j < num_coords; ++j) {
      atomicAdd(&centroids[centr_idx*num_coords + j], dev_points[i*num_coords + j]);
    }
  }

#ifdef SHARED_MEM
  for (int i = point_id; i < num_centroids; i+=stride) {
    for (int j = 0; j < num_coords; ++j) {
      atomicAdd(&dev_centroids[i*num_coords + j], shared_centroids[i*num_coords + j]);
    }
  }
#endif
}

__device__ static void find_nearest_cluster(double *dev_points,
                                            double *dev_centroids,
                                            int *dev_cluster, int num_points,
                                            int num_coords, int num_centroids) {
  int point_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (point_id >= num_points)
    return;

  // start with the dist between the point and the first centroid
  double min_dist =
      euclidian_dist_squared(dev_points, dev_centroids, num_points,
                             num_centroids, num_coords, point_id, 0);
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

  dev_cluster[point_id] = min_index;
}

__global__ static void assign_and_accum_centroids(double *dev_points,
                                            double *dev_centroids, double *dev_old_centroids,
                                            int *dev_cluster, int num_points,
                                            int num_coords, int num_centroids) {
  find_nearest_cluster(dev_points, dev_old_centroids, dev_cluster, num_points, num_coords, num_centroids);
  accum_centroids(dev_points, dev_centroids, dev_cluster, num_points, num_coords, num_centroids);
}


__global__ static void compute_converged(double *centroids,
                                      double *old_centroids, int num_centroids,
                                      int num_coords, int shared_length) {
  extern __shared__ bool shared[]; // array of bools in shared memory

  int centroid_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (centroid_id >= num_centroids)
    return;

  shared[centroid_id] = true;
  for (int i = 0; i < num_coords; ++i) {
    if (fabs(centroids[centroid_id*num_coords + i] - old_centroids[centroid_id*num_coords + i])) {
        shared[centroid_id] = false;
    }
  }

  // sync threads here to fill shared with the correct data
  __syncthreads();

  // to do the reduction, shared_length has to be a power of 2
  for (int s = shared_length / 2; s > 0; s >>= 1) {
    if (centroid_id < s) {
      shared[centroid_id] &= shared[centroid_id + s];
      //printf("reducing %d + %d to %d\n", centroid_id, s, shared[centroid_id]);
    }
    __syncthreads();
  }

  if (centroid_id == 0) {
    converged = shared[0];
    //printf("converged is %d\n", shared[0]);
  }
}

/*
 * kmeans
 */

double **kmeans(double **const points, double **centroids,
                double **old_centroids, int num_points, int num_coords,
                int num_centroids, int *const cluster, int *cluster_size,
                int *num_iterations, int max_iterations, double threshold, double *time_elapsed) {

  double *dev_points;
  double *dev_centroids;
  double *dev_old_centroids;
  int *dev_cluster;

  double *flat_points =
      (double *)malloc(num_points * num_coords * sizeof(double));
  double *flat_centroids =
      (double *)malloc(num_centroids * num_coords * sizeof(double));
  double *flat_old_centroids =
      (double *)malloc(num_centroids * num_coords * sizeof(double));

  int iterations = 0;

  for (int i = 0; i < num_points; ++i)
    cluster[i] = -1; // init cluster membership to default val

  for (int i = 0; i < num_centroids; ++i)
    cluster_size[i] = 0;

  const size_t threads_per_block = 128; // This is simply a design decision
  const size_t num_blocks =
      (num_points + threads_per_block - 1) / threads_per_block;
  const size_t shared_mem_per_block = num_centroids * num_coords * sizeof(double);

  // this is the next power of 2 after num_centroids
  // for use in compute_converged()
  const size_t reduction_threads = powf(2, ceil(log(num_centroids)/log(2)));
  const size_t shared_mem_comparison = reduction_threads * sizeof(bool);

  cudaSetDevice(0);
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

#ifdef DEBUG
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
	printf("Device Number: %d\n", 0);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (KHz): %d\n",
         prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n",
         prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
         2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
#endif

  cudaMalloc((void **)&dev_points, num_points * num_coords * sizeof(double));
  cudaCheckError("malloc dev_points");

  cudaMalloc((void **)&dev_centroids,
             num_centroids * num_coords * sizeof(double));
  cudaCheckError("malloc dev_centroids");
  
  cudaMalloc((void **)&dev_old_centroids,
             num_centroids * num_coords * sizeof(double));
  cudaCheckError("malloc dev_old_centroids");

  cudaMalloc((void **)&dev_cluster, num_points * sizeof(int));
  cudaCheckError("malloc dev_cluster");

  // flatten the 2D points array to copy it onto the GPU
  flatten_2D_array(points, flat_points, num_points, num_coords);
  cudaMemcpy(dev_points, flat_points, num_points * num_coords * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaCheckError("copy points to device");

  // cluster is already a 1D array so it doesn't need to be flattened
  cudaMemcpy(dev_cluster, cluster, num_points * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaCheckError("copy cluster to device");

  flatten_2D_array(centroids, flat_centroids, num_centroids, num_coords);
  flatten_2D_array(old_centroids, flat_old_centroids, num_centroids, num_coords);

  clock_t start = clock();

  do {
    cudaMemcpy(dev_old_centroids, flat_old_centroids,
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError("copy flat_old_centroids into dev_old_centroids");
    /*cudaMemcpy(dev_centroids, flat_centroids,
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError("copy flat_centroids into dev_centroids");
    */
    cudaMemset(dev_centroids, 0, num_centroids * num_coords * sizeof(double));
    cudaCheckError("memset centroids");


    // for each point in dev_points, finds the nearest centroid from 
    // dev_centroids, and stores the index in dev_cluster
    assign_and_accum_centroids<<<num_blocks, threads_per_block,
                           shared_mem_per_block>>>(dev_points, dev_centroids, dev_old_centroids,
                                                   dev_cluster, num_points,
                                                   num_coords, num_centroids);

    // while we are waiting for the GPU kernel, clear centroids
    for (int i = 0; i < num_centroids; ++i)
      for (int j = 0; j < num_coords; ++j)
        centroids[i][j] = 0.0;

    // synchronize here so that we can ensure dev_cluster has been filled
    cudaDeviceSynchronize();
    cudaCheckError("synchronize after assign_and_accum");

    cudaMemcpy(cluster, dev_cluster, num_points * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaCheckError("copy point->cluster map back to host");
    cudaMemcpy(flat_centroids, dev_centroids, num_centroids * num_coords * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaCheckError("copy centroids back to host");

    unflatten_1D_array(flat_centroids, centroids, num_centroids, num_coords);

    for (int i = 0; i < num_points; ++i) {
      // get which cluster the point belongs to 
      int cluster_idx = cluster[i];
      assert(cluster_idx >= 0 && cluster_idx < num_centroids);

      // increment cluster_size at the correct index
      ++cluster_size[cluster_idx];
      assert(cluster_size[cluster_idx] < num_points);
    }

    // scale each centroid's sum by 1/cluster_size
    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        // ensure that we don't divide by 0
        if (cluster_size[i] > 0) {
          centroids[i][j] /= cluster_size[i];
        } else
          assert(0);
      }
      // reset the cluster_size at each index
      cluster_size[i] = 0;
    }

    // flatten the newly calculated
    flatten_2D_array(centroids, flat_centroids, num_centroids, num_coords);
    cudaMemcpy(dev_centroids, flat_centroids,
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError("copy flat_centroids into dev_centroids after calculation");

    cudaMemcpy(dev_old_centroids, flat_old_centroids,
               num_centroids * num_coords * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError("copy flat_old_centroids into dev_old_centroids");

    // test if each centroid has converged
    // if one hasn't conv, puts false in the converged __device__ var
    compute_converged<<<1, reduction_threads, shared_mem_comparison>>>(dev_centroids, dev_old_centroids, num_centroids, num_coords, shared_mem_comparison);
    cudaDeviceSynchronize();

    bool break_flag = false;
    cudaMemcpyFromSymbol(&break_flag, converged, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    if (break_flag)
      break;
    

    // deep copy from centroids into old_centroids (and their flat versions)
    for (int i = 0; i < num_centroids; ++i) {
      for (int j = 0; j < num_coords; ++j) {
        old_centroids[i][j] = centroids[i][j];
        flat_old_centroids[i*num_coords + j] = flat_centroids[i*num_coords + j];
      }
    }

  } while (++iterations < max_iterations);

  *time_elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;

  cudaFree(dev_points);
  cudaFree(dev_centroids);
  cudaFree(dev_old_centroids);
  cudaFree(dev_cluster);

  free(flat_points);
  free(flat_centroids);
  free(flat_old_centroids);

  *num_iterations = iterations;
  return centroids;
}

/*
 * Utility Functions
 */

void cudaCheckError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void flatten_2D_array(double **src, double *dest, int r, int c) {
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j)
      dest[i * c + j] = src[i][j];
}

void unflatten_1D_array(double *src, double **dest, int r, int c) {
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j)
      dest[i][j] = src[i*c + j];
}
