#include "kmeans.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>

using namespace std;

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
      points[i][j] = points_vec[0][j];
  }
  return points;
}
