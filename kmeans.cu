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
 * Global Vars
 */
bool break_flag;
bool mutex;

pthread_mutex_t lock;
pthread_spinlock_t slock;
pthread_barrier_t barrier;

/*
 * Function Prototypes
 */
void readFile(vector<Point>& ds, string filePath) {


int main(int argc, char *argv[]) {
  int clusters;
  int max_iterations;
  double threshold;
  int workers;
  string input;
  vector<int> numPointsPerCentroid;

  threshold = DEFAULT_THRESH;
  max_iterations = INT_MAX;

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

    case 'l':
      mutex = false;
      break;
    }
  }

  vector<Point> dataSet; 
  readFile(dataSet, input);
  int numPoints = dataSet.size();

  cout << numPoints << endl;

  //can't use Point anymore because it's vector-backed
  Point *centroids = (Point *) malloc(clusters * sizeof(Point));
  Point *old_centroids = randomCentroids(dataSet, clusters, *dataSet);

  start = clock();

  kmeans();

  duration = (clock() - start) / (double) CLOCKS_PER_SEC;

  cout << duration << endl;

  printPointVector(dataSet);
  printPointVector(centroids);

  delete dataSet;
}

void kmeans() {

}


void readFile(vector<Point>& ds, string filePath) {
  vector<Point> points;

  ifstream inFile;
  inFile.open(filePath);

  if (!inFile) {
    cerr << "Unable to open input file";
    exit(1); // call system to stop
  }

  int size;
  inFile >> size;

#ifdef DEBUG
  cout << "Size: " << size << endl;
#endif

  string line;
  getline(inFile, line);
  while (size--) {
    getline(inFile, line);
    vector<double> nums;
    istringstream is(line);

    int lineNumber;
    is >> lineNumber;

    assert(points.size() == lineNumber - 1);

    double num;
    while (is >> num)
      nums.push_back(num);

    Point point(nums);
    points.push_back(point);
  }

  inFile.close();

  copy(begin(points), end(points), begin(ds));
}

void print_help() {
  cout << "Format: " << endl
       << "kmeans -c clusters -t threshold -i iterations -w workers -I "
          "path/to/input"
       << endl;
}

void printPointVector(const vector<Point> &points) {
  for (const auto &point : points) {
    for (int i = 0; i < point.vals.size() - 1; ++i) {
      double val = point.vals[i];
      cout << val << ", ";
    }
    cout << point.vals[point.vals.size() - 1] << endl;
  }
}

vector<vector<Point>> splitDataSet(const vector<Point>& points, int workers) {
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
