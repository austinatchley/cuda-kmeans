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

#include <cuda_runtime_api.h>
#ifdef PROFILE
#include <cuda_profiler_api.h>
#endif

/*
 * Function Prototypes
 */

void kmeans(
    Point *points,
    Point *centroids,
    Point *old_centroids,
    int clusters,
    int max_iterations,
    double threshold,
    int workers);


Point *random_centroids(const Point *points, int num_points, int clusters);
vector<vector<Point>> split_data_set(const vector<Point>& points, int workers);

void print_point_vector(const vector<Point> &points);
void read_file(vector<Point>& ds, string filePath);
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

    case 'w':
      assert(optarg);
      workers = atoi(optarg);
      break;

    case 'I':
      assert(optarg);
      input = optarg;
      break;
    }
  }

  vector<Point> data_set; 
  read_file(data_set, input);
  int num_points = data_set.size();

  Point *points = (Point *) malloc(num_points * sizeof(Point));
  copy(begin(data_set), end(data_set), points);

  cout << num_points << endl;

  Point *centroids = (Point *) malloc(clusters * sizeof(Point));
  Point *old_centroids = random_centroids(points, num_points, clusters);

  clock_t start = clock();

  kmeans(points, centroids, old_centroids, clusters, max_iterations, threshold, workers);

  clock_t duration = (clock() - start) / (double) CLOCKS_PER_SEC;

  cout << duration << endl;

  print_point_vector(data_set);
  //print_point_vector(centroids);
}

void kmeans(
    Point *points,
    Point *centroids,
    Point *old_centroids,
    int clusters,
    int max_iterations,
    double threshold,
    int workers) {
  cout << "kmeans" << endl;
}

Point *random_centroids(const Point *points, int num_points, int num_clusters) {
  srand(time(NULL));
  vector<int> indicesUsed;
  Point *centroids = (Point *) malloc(num_points * sizeof(Point));

  for (int i = 0; i < num_points; ++i) {
    int index;

    // Generate rand index that hasn't been used
    do {
      index = ((int)rand()) % num_points;
    } while (find(begin(indicesUsed), end(indicesUsed), index) !=
             end(indicesUsed));
    indicesUsed.push_back(index);

    Point p(points[index].vals);
    centroids[index] = p;
  }
  return centroids;
}

vector<vector<Point>> split_data_set(const vector<Point>& points, int workers) {
  vector<vector<Point>> sets;

  int size = points.size();
  for (int i = 0; i < workers; ++i) {
    vector<Point> set;

    int begin = ((size * i) / workers);
    int end = ((size * (i + 1)) / workers);

    for (int i = begin; i < end; ++i)
      set.push_back(points[i]);

#ifdef DEBUG
    cout << workers << " workers" << endl;
    cout << "Begin at " << ((size * i) / workers) << endl;
    cout << "End at " << ((size * (i + 1)) / workers) << endl;
#endif
    sets.push_back(set);
  }
  return sets;
}

void print_help() {
  cout << "Format: " << endl
       << "kmeans -c clusters -t threshold -i iterations -w workers -I "
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

