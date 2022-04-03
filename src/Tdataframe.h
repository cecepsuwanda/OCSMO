#include <mutex>
#include "Tread_file.h"
#include "Tmy_dttype.h"
#include "Tlabel_stat.h"
#include "Tbelow_above.h"
#include "Tbase_dataframe.h"
#include <thread>
#include <unordered_map>
#include <map>
#include <cmath>
#include <string>

using namespace std;

#ifndef Included_Tdataframe_H

#define Included_Tdataframe_H

class Tdataframe : public Tbase_dataframe
{
private:

  map<int, int> _unique_attr;
  Tlabel_stat _stat_label;
  Tmap_col_split _map_col_split;
  int _idx_label;
  bool is_non_continuous = false;
  bool is_42 = false;
  //bool _search_uniqe_val_on = true;
  Tconfig* config;
  vector<string> _list_label;

  void calculate_metric(map<Tmy_dttype, Tlabel_stat>* _col_pot_split, float & current_overall_metric, string & split_value, Tlabel_stat & stat_label);

  void handle_continuous(int idx, float & current_overall_metric, string & split_value);
  void handle_non_continuous(int idx, float & current_overall_metric, string & split_value);

public:
  Tdataframe();
  Tdataframe(Tconfig* v_config);
  ~Tdataframe();

  Tdataframe(const Tdataframe &t)
  {
    //cout << "Copy Constructor" << endl;
    _data = t._data;
    _data_header = t._data_header;
    _data_type = t._data_type;
    _filter = t._filter;

    _unique_attr = t._unique_attr;

    _jml_col = t._jml_col;
    _jml_row = t._jml_row;

    _stat_label = t._stat_label;
    _idx_label = t._idx_label;

    _nm_file = t._nm_file;

    is_non_continuous = t.is_non_continuous;
    is_42 = t.is_42;
    _jml_total_row = t._jml_total_row;
    config = t.config;
     
     _stat_label.set_config(config);
     _map_col_split.set_config(config);

  }


  Tdataframe& operator = (const Tdataframe &t)
  {
    //cout << "operator =" << endl;
    this->_data = t._data;
    this->_data_header = t._data_header;
    this->_data_type = t._data_type;
    this->_filter = t._filter;
    this->_unique_attr = t._unique_attr;
    this->_jml_col = t._jml_col;
    this->_jml_row = t._jml_row;
    this->_stat_label = t._stat_label;
    this->_idx_label = t._idx_label;
    this->_nm_file = t._nm_file;
    this->is_non_continuous = t.is_non_continuous;
    this->is_42 = t.is_42;
    this->_jml_total_row = t._jml_total_row;
    this->config = t.config;

    this->_stat_label.set_config(this->config);
    this->_map_col_split.set_config(this->config);

    return *this;
  }

  void set_shared_memory_on();
  void set_shared_memory_off();

  //void set_search_uniqe_val_off();

  void read_data_type(string nm_f);

  void stat_tabel();
  map<string, int> get_stat_label();
  float get_estimate_error();
  string get_max_label();
  bool is_single_label();
  map<int, int> get_unique_attr();

  int getjmlcol_svm();
  int getjmlrow_svm();
  vector<string> get_record_svm();  
  vector<vector<string>> get_all_record_svm();

  map<Tmy_dttype, Tlabel_stat>* get_col_split(int idx);
  void clear_col_split();

  // void set_config(Tconfig v_config);

  int get_jml_valid_attr();
  int get_valid_attr(int idx);

  void info();

  void add_filter(int idx_col, int idx_opt, string value);
  void add_filter(field_filter filter);
  void ReFilter();

  void clear_map_col_split();

  void split_data(int split_column, string split_value, Tdataframe &data_below, Tdataframe &data_above);
  float get_entropy();

  string get_nm_header(int idx_col);
  int get_opt(int idx_col, int is_below);
  void get_col_pot_split(int idx);
  void calculate_overall_metric(int idx, float &current_overall_metric, string &split_value);  

  vector<string> goto_rec(int idx);
  vector<string> get_list_label();

};

#endif