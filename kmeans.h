#ifndef KMEANS
#define KMEANS

#define DEFAULT_THRESH 0.0000001f

#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

/*
 * Classes
 */

class Point {
public:
  vector<double> vals;

  Point(vector<double> newVals) : vals(newVals) {}

  int getDimensions() { return vals.size(); }

  bool operator<(const Point &other) const { return this->vals < other.vals; }

  bool operator>(const Point &other) const { return this->vals > other.vals; }

  bool operator==(const Point &other) const { return this->vals == other.vals; }

  bool operator!=(const Point &other) const { return this->vals != other.vals; }

  Point &operator+(const Point &other) {
    for (int i = 0; i < vals.size(); ++i)
      this->vals[i] += other.vals[i];

    return *this;
  }

  Point &operator/(int val) {
    const int len = vals.size();
    for (int i = 0; i < len; ++i)
      this->vals[i] = this->vals[i] / val;

    return *this;
  }

  double &operator[](int i) { return vals[i]; }
};

/*
 * Namespaces
 */

namespace point {
typedef std::map<Point, int> pointMap;
}

/*
 * Function Prototypes
 */

double **kmeans(double **points, double **centroids, double **old_centroids,
                int num_points, int num_coords, int num_centroids, int *cluster,
                int *cluster_size, int *num_iterations, int max_iterations,
                double threshold, double *time_elapsed);

double **read_file(vector<Point> &ds, string file_path, int *size, int *coords);
#endif
