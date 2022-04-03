#include <mutex>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include "Twrite_file.h"

using namespace std;

#ifndef Included_Tconf_metrix_H

#define Included_Tconf_metrix_H

struct Tdata
{
  int TP, FP, TN, FN, jml;
  float  accuracy,recall, precision,f1;
};

class Tconf_metrix
{
private:
  map<string, map<string, int>> matrik;
  map<string, Tdata> matrik1;
  float accuracy, precision, recall,f1;
  int failed, tepat, tdk_tepat, jml_data;
  map<string, int> label;
  map<string, string> konversi_asli;
  map<string, string> konversi_tebakan;

  mutable std::mutex v_mutex;

public:
  Tconf_metrix();
  ~Tconf_metrix();
  
  void add_jml(string asli, string tebakan, int jml);
  void kalkulasi();
  int get_TP(string kelas);
  int get_TN(string kelas);
  int get_FP(string kelas);
  int get_FN(string kelas);
  float get_F1();
  void save(string nm_file, string param_nm_file, int param_depth, int param_min_sample, double param_gamma, double param_nu, double param_credal_s);

  void add_konversi_asli(string dari,string ke);
  void add_konversi_tebakan(string dari,string ke);

  friend ostream & operator << (ostream &out, const Tconf_metrix &tc);
};

#endif