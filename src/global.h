#include <stdarg.h>
#include <string>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <cmath>

using namespace std;

#ifndef Included_global_H

#define Included_global_H

struct Tconfig
{
  string f_datatype = "";
  string f_train = "";
  string f_test = "";
  string path_model = "";
  string svm_path = "";

  bool save_train = false;
  bool save_test = false;
  bool use_credal = false;
  double credal_s = 0.0;
  bool limited = false;

  bool train_svm = false;
  bool feature_selection = false;
  bool normal_only = false;

  double gamma = 0.0;
  double V = 0.0;
  double eps = 1.0;

  int depth = 0;
  int min_sample = 0;
  int threshold = 0;

  bool search_uniqe_val = false;
};

struct Tmetric_split_value
{
  int idx = 0;
  float overall_metric = -1;
  string split_value = "-1";
};

class Node
{
public:
  int criteriaAttrIndex;
  string attrValue;
  string label;

  int treeIndex;
  bool isLeaf;
  int opt;
  int idx_svm;

  vector<int> children;

  Node()
  {
    criteriaAttrIndex = -1;
    attrValue = "-1";
    label = "-1";
    treeIndex = -1;
    isLeaf = false;
    opt = -1;
    idx_svm = -1;
  }
};

static void cetak_stdout(const char *s)
{
  fputs(s, stdout);
  fflush(stdout);
}

static void cetak(const char *format, ...)
{
  char buffer[256];
  va_list args;
  va_start(args, format);
  vsprintf(buffer, format, args);
  // perror (buffer);
  va_end(args);
  cetak_stdout(buffer);
}

static double bulat_nol(double val, double tolerance, int digit)
{
  double tmp = val;

  if (abs(val) < tolerance)
  {
    tmp = 0.0;
  }
  else
  {
    const double multiplier = std::pow(10.0, digit);
    tmp = ceil(val * multiplier) / multiplier;
  }
  // double tmp = val;
  return tmp;
}

static bool AlmostEqualRelative(double A, double B,
                                float maxRelDiff = FLT_EPSILON)
{
  // Calculate the difference.
  double diff = fabs(A - B);
  A = fabs(A);
  B = fabs(B);
  // Find the largest
  double largest = (B > A) ? B : A;

  if (diff <= largest * maxRelDiff)
    return true;
  return false;
}

#endif