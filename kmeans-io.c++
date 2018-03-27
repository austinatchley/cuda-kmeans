#include "kmeans.h"
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <map>
#include <memory>

using namespace std;

void read_file(vector<Point>& ds, string file_path) {
  vector<Point> points;

  ifstream in_file;
  in_file.open(file_path);

  if (!in_file) {
    cerr << "Unable to open input file";
    exit(1); // call system to stop
  }

  int size;
  in_file >> size;

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

    assert(points.size() == line_num - 1);

    double num;
    while (is >> num)
      nums.push_back(num);

    Point point(nums);
    points.push_back(point);
  }

  in_file.close();

  copy(begin(points), end(points), begin(ds));
}
