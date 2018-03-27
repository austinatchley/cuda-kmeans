#include "kmeans.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <map>
#include <memory>
#include <climits>

/*
 * Function Prototypes
 */

void kmeans(
    double **points,
    double **centroids,
    double **old_centroids,
    int num_points,
    int num_coords,
    int num_centroids,
    int max_iterations,
    double threshold);


double **random_centroids(double **points, int num_points, int num_clusters, int num_coords);
void print_point_vector(const vector<Point> &points);
void print_help();


int main(int argc, char *argv[]) {
  int clusters;
  double threshold = DEFAULT_THRESH;
  int max_iterations = INT_MAX;
  int workers;
  string input;
  vector<int> num_pointsPerCentroid;


  int opt, val;
  while ((opt = getopt(argc, argv, "hc:t:i:w:I:l")) != -1) {
    switch (opt) {
    case 'h':
      print_help();
      exit(0);
      break;

    case 'c':
      assert(optarg);
      clusters = atoi(optarg);
      break;

    case 't':
      assert(optarg);
      threshold = atof(optarg);
      break;

    case 'i':
      assert(optarg);
      val = atoi(optarg);
      if (val != 0)
        max_iterations = val;
      break;

    case 'I':
      assert(optarg);
      input = optarg;
      break;
    }
  }

  vector<Point> data_set_vec; 
  int num_points, num_coords;
  double **points = read_file(data_set_vec, input, &num_points, &num_coords);

  cout << num_points << endl;

  double **centroids = (double **) malloc(clusters * sizeof(double *));
  double **old_centroids = random_centroids(points, num_points, clusters, num_coords);

  clock_t start = clock();

  kmeans(points, centroids, old_centroids, num_points, num_coords, clusters, max_iterations, threshold);

  clock_t duration = (clock() - start) / (double) CLOCKS_PER_SEC;

  cout << duration << endl;

  //print_point_vector(centroids);
}



double **random_centroids(double **points, int num_points, int num_clusters, int num_coords) {
  srand(time(NULL));
  vector<int> indicesUsed;
  double **centroids = (double **) malloc(num_clusters * sizeof(float *));

  for (int i = 0; i < num_points; ++i) {
    int index;

    // Generate rand index that hasn't been used
    do {
      index = ((int)rand()) % num_points;
    } while (find(begin(indicesUsed), end(indicesUsed), index) !=
             end(indicesUsed));
    indicesUsed.push_back(index);

    centroids[index] = (double *) malloc(num_coords * sizeof(double));
    copy(points[index], points[index] + num_coords, centroids[index]);
  }
  return centroids;
}

void print_help() {
  cout << "Format: " << endl
       << "kmeans -c clusters -t threshold -i iterations -I file"
          "path/to/input"
       << endl;
}

void print_point_vector(const vector<Point> &points) {
  for (const auto &point : points) {
    for (int i = 0; i < point.vals.size() - 1; ++i) {
      double val = point.vals[i];
      cout << val << ", ";
    }
    cout << point.vals[point.vals.size() - 1] << endl;
  }
}

