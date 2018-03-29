#include "kmeans.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

/*
 * Function Prototypes
 */
double **random_centroids(double **points, int num_points, int num_centroids,
                          int num_coords);
void print_point_vector(const vector<Point> &points);
void print_point_array(double **const points, const int num_points,
                       const int num_coords);
void print_help();

int main(int argc, char *argv[]) {
  int num_centroids;
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
      num_centroids = atoi(optarg);
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

  double **centroids =
      random_centroids(points, num_points, num_centroids, num_coords);

  double **old_centroids = (double **)malloc(num_centroids * sizeof(double *));
  for (int i = 0; i < num_centroids; ++i) {
    old_centroids[i] = (double *)malloc(num_coords * sizeof(double));
    for (int j = 0; j < num_coords; ++j)
      old_centroids[i][j] = centroids[i][j];
  }

  int *cluster = (int *)malloc(num_points * sizeof(int));
  int *cluster_size = (int *)malloc(num_centroids * sizeof(int));

  int *iterations = (int *)malloc(sizeof(int));
  double *duration = (double *)malloc(sizeof(double));

#ifdef DEBUG
  clock_t begin_time = clock();
#endif

  double **final_centroids = kmeans(
      points, centroids, old_centroids, num_points, num_coords, num_centroids,
      cluster, cluster_size, iterations, max_iterations, threshold, duration);

  cout << *iterations << endl;
  cout << *duration << endl;

#ifdef DEBUG
  double end_to_end = (clock() - begin_time) / (double)CLOCKS_PER_SEC;
  cout << end_to_end << endl;
#endif

  print_point_array(points, num_points, num_coords);
  print_point_array(final_centroids, num_centroids, num_coords);

  // print_point_vector(centroids);
}

double **random_centroids(double **points, int num_points, int num_centroids,
                          int num_coords) {
  vector<int> indices_used;
  double **centroids = (double **)malloc(num_centroids * sizeof(double *));
  srand(time(NULL));

  for (int i = 0; i < num_centroids; ++i) {
    centroids[i] = (double *)malloc(num_coords * sizeof(double));
    int index;

    // Generate rand index that hasn't been used
    do {
      index = ((int)rand()) % num_points;
    } while (find(begin(indices_used), end(indices_used), index) !=
             end(indices_used));
    indices_used.push_back(index);

    copy(&points[index][0], &points[index][num_coords], centroids[i]);
  }

  return centroids;
}

void print_help() {
  cout << "Format: " << endl
       << "kmeans -c num_centroids -t threshold -i iterations -I file"
          "path/to/input"
       << endl;
}

void print_point_array(double **const points, const int num_points,
                       const int num_coords) {
  for (int i = 0; i < num_points; ++i) {
    for (int j = 0; j < num_coords - 1; ++j) {
      double val = points[i][j];
      cout << val << ", ";
    }
    cout << points[i][num_coords - 1] << endl;
  }
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

double **read_file(vector<Point> &ds, string file_path, int *num_points,
                   int *num_coords) {
  vector<Point> points_vec;

  ifstream in_file;
  in_file.open(file_path);

  if (!in_file) {
    cerr << "Unable to open input file";
    exit(1); // call system to stop
  }

  size_t size = 0;
  in_file >> size;
  *num_points = size;

#ifdef DEBUG
  cout << "Size: " << size << endl;
#endif

  string line;
  getline(in_file, line);
  while (size--) {
    getline(in_file, line);
    vector<double> nums;
    istringstream is(line);

    int line_num;
    is >> line_num;

    assert(points_vec.size() == line_num - 1);

    double num;
    while (is >> num)
      nums.push_back(num);

    Point point(nums);
    points_vec.push_back(point);
  }

  in_file.close();

  *num_coords = points_vec[0].getDimensions();
  double **points = (double **)malloc(points_vec.size() * sizeof(double *));
  for (int i = 0; i < points_vec.size(); ++i) {
    points[i] = (double *)malloc(*num_coords * sizeof(double));
    for (int j = 0; j < *num_coords; ++j)
      points[i][j] = points_vec[i][j];
  }
  return points;
}
