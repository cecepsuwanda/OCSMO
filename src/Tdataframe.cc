
#include "Tdataframe.h"
#include <iterator>


Tdataframe::Tdataframe()
{
  _idx_label = -1;
}

Tdataframe::Tdataframe(Tconfig* v_config)
{
  _idx_label = -1;
  config = v_config;
  _stat_label.set_config(config);
  _map_col_split.set_config(config);
}

Tdataframe::~Tdataframe()
{
  _stat_label.clear();
  _list_label.clear();
}

// void Tdataframe::set_search_uniqe_val_off()
// {
//   _search_uniqe_val_on = false;
// }

void Tdataframe::set_shared_memory_on()
{
  _data.set_shared_memory_on();
}

void Tdataframe::set_shared_memory_off()
{
  _data.set_shared_memory_off();
}


void Tdataframe::read_data_type(string nm_f)
{

  Tread_file tmp;
  tmp.setnm_f(nm_f, ": ");

  //cout << "Tdataframe read_data_type setnm_f " << endl;

  bool is_label = false;

  int i = 0;
  while ((!tmp.is_eof()) and (!is_label))
  {
    _data_header.push_back(tmp.get_col_val(0));
    _data_type.push_back(tmp.get_col_val(1));

    if (tmp.get_col_val(0) == "label")
    {
      _idx_label = i;
      is_label = true;
    }

    if ((tmp.get_col_val(0) != "label") and (tmp.get_col_val(1) != "continuous."))
    {
      is_non_continuous = true;
    }

    //cout << "Tdataframe read_data_type next_record " << endl;
    tmp.next_record();
    i++;
  }
  _jml_col = i;
  tmp.close_file();

  if (is_non_continuous)
  {
    is_42 = _jml_col == 42;
  }


  stat_tabel();
}

void Tdataframe::stat_tabel()
{
  _stat_label.clear();

  _data.index_on();
  if ((_filter.size() > 0) or (_by_pass_filter==true)) {
    _data.clear_index();
  } else {
    _data.index_off();
  }

  int i = 0;
  _list_label.clear();

  _data.reset_file();
  while (!_data.is_eof())
  {

    if (is_pass())
    {
      if ((_filter.size() > 0) or (_by_pass_filter==true)) {
        _data.add_index();
      }

      _stat_label.add(_data.get_col_val(_idx_label));
      _list_label.push_back(_data.get_col_val(_idx_label));

      if (config->search_uniqe_val)
      {
        while (!_data.is_end_col())
        {
          if (_data.get_idx_col() < _idx_label) {
            _map_col_split.add_data(_data.get_idx_col(), _data.get_col_val(), _data_type[_data.get_idx_col()], _data.get_col_val(_idx_label));
          }
          _data.next_col();
        }
      }

      i++;
    }
    _data.next_record();
  }

  if ((_filter.size() > 0) or (_by_pass_filter==true)) {
    _data.clear_memory();
    _data.save_to_memory();
    _data.clear_index();
  }

  _data.index_on();
  _jml_row = i;

  if (config->search_uniqe_val)
  {
    _map_col_split.cek_valid_attr(_jml_row);
  }

}

map<string, int> Tdataframe::get_stat_label()
{
  return _stat_label.get_map();
}

float Tdataframe::get_estimate_error()
{
  return _stat_label.get_estimate_error();
}

string Tdataframe::get_max_label()
{
  return _stat_label.get_max_label();
}

bool Tdataframe::is_single_label()
{
  return _stat_label.is_single_label();
}

map<int, int> Tdataframe::get_unique_attr()
{
  return _unique_attr;
}

int Tdataframe::getjmlcol_svm()
{
  if (is_non_continuous)
  {
    if (config->feature_selection)
    {
      int jml = 0;
      for (auto itr = _unique_attr.begin(); itr != _unique_attr.end(); ++itr)
      {
        switch (itr->first) {
        case 1:
          jml = jml + 3;
          break;
        case 2:
          jml = jml + 2;
          break;
        case 3:
          jml = jml + 2;
          break;
        default:
          jml = jml + 1;
          break;
        }
      }
      return jml + 1;

    } else {
      return  (is_42 ? 46 : 33);
    }
  } else {
    if (config->feature_selection)
    {
      return _unique_attr.size() + 1;
    } else {
      return _jml_col;
    }
  }
}

int Tdataframe::getjmlrow_svm()
{
  if (config->normal_only)
  {
    return _stat_label.get_value("normal");
  } else {
    return _jml_row;
  }
}

vector<string> Tdataframe::get_record_svm()
{
  if (!is_non_continuous)
  {
    if (config->feature_selection) {

      vector<string> vec;
      for (auto itr = _unique_attr.begin(); itr != _unique_attr.end(); ++itr)
      {
        vec.push_back(_data.get_col_val(itr->first));
      }

      vec.push_back(_data.get_col_val(_idx_label));

      return vec;

    } else {
      return _data.get_record();
    }

  } else {

    if (config->feature_selection) {

      vector<string> vec;
      for (auto itr = _unique_attr.begin(); itr != _unique_attr.end(); ++itr)
      {
        switch (itr->first) {
        case 1:
          vec.push_back((_data.get_col_val(itr->first) == "tcp" ? "1" : "0" ));
          vec.push_back((_data.get_col_val(itr->first) == "udp" ? "1" : "0" ));
          vec.push_back((_data.get_col_val(itr->first) == "icmp" ? "1" : "0" ));
          break;
        case 2:
          vec.push_back(((_data.get_col_val(itr->first) == "private") or (_data.get_col_val(itr->first) == "ecri") or (_data.get_col_val(itr->first) == "ecr_i") or (_data.get_col_val(itr->first) == "http")) ? "0" : "1");
          vec.push_back(((_data.get_col_val(itr->first) == "private") or (_data.get_col_val(itr->first) == "ecri") or (_data.get_col_val(itr->first) == "ecr_i") or (_data.get_col_val(itr->first) == "http")) ? "1" : "0");
          break;
        case 3:
          vec.push_back((_data.get_col_val(itr->first) == "SF") ? "0" : "1");
          vec.push_back((_data.get_col_val(itr->first) == "SF") ? "1" : "0");
          break;
        default:
          vec.push_back(_data.get_col_val(itr->first));
          break;
        }
      }

      vec.push_back(_data.get_col_val(_idx_label));

      return vec;
    } else {

      vector<string> vec;//, tmp_data = _data.get_record();

      for (int i = 0; i < _jml_col; ++i)
      {

        switch (i) {
        case 1:
          vec.push_back((_data.get_col_val(i) == "tcp" ? "1" : "0" ));
          vec.push_back((_data.get_col_val(i) == "udp" ? "1" : "0" ));
          vec.push_back((_data.get_col_val(i) == "icmp" ? "1" : "0" ));
          break;
        case 2:
          vec.push_back(((_data.get_col_val(i) == "private") or (_data.get_col_val(i) == "ecri") or (_data.get_col_val(i) == "ecr_i") or (_data.get_col_val(i) == "http")) ? "0" : "1");
          vec.push_back(((_data.get_col_val(i) == "private") or (_data.get_col_val(i) == "ecri") or (_data.get_col_val(i) == "ecr_i") or  (_data.get_col_val(i) == "http")) ? "1" : "0");
          break;
        case 3:
          vec.push_back((_data.get_col_val(i) == "SF") ? "0" : "1");
          vec.push_back((_data.get_col_val(i) == "SF") ? "1" : "0");
          break;
        default:
          vec.push_back(_data.get_col_val(i));
          break;
        }

      }

      //tmp_data.clear();
      //tmp_data.shrink_to_fit();

      return vec;
    }
  }
}

vector<vector<string>> Tdataframe::get_all_record_svm()
{
  //std::lock_guard<std::mutex> lock(v_mutex);
  ReFilter();
  if(config->search_uniqe_val){
    clear_col_split();
  }

  vector<vector<string>> Table;

  vector<string> tmp_data;

  _data.reset_file();
  while (!_data.is_eof())
  {
    //cout << " get_all_record_svm get_record_svm " << endl;
    tmp_data = get_record_svm();

    bool is_pass = (config->normal_only ? (tmp_data[tmp_data.size() - 1].compare("normal") == 0) : true);
    if (is_pass) {
      Table.push_back(tmp_data);
    }
    //cout << " get_all_record_svm next_record " << endl;
    _data.next_record();
  }

  clear_memory();

  return Table;
}

vector<string> Tdataframe::goto_rec(int idx)
{
  //ReFilter();
  
  vector<string> tmp_data;
  _data.reset_file();
  _data.goto_rec(idx);
  tmp_data = get_record_svm();
  tmp_data.erase(tmp_data.begin()+_idx_label);
  return tmp_data;

}

map<Tmy_dttype, Tlabel_stat>* Tdataframe::get_col_split(int idx)
{
  return _map_col_split.get_pot_split(idx);
}


void Tdataframe::clear_col_split()
{
  _map_col_split.clear();
}

// void Tdataframe::set_config(Tconfig v_config)
// {
//   config = v_config;
//   _stat_label.set_config(config);
//   _map_col_split.set_config(config);
// }

int Tdataframe::get_jml_valid_attr()
{
  return _map_col_split.get_jml_valid_attr();
}

int Tdataframe::get_valid_attr(int idx)
{
  return _map_col_split.get_valid_attr(idx);
}

void Tdataframe::add_filter(int idx_col, int idx_opt, string value)
{
  field_filter f;
  f.idx_col = idx_col;
  f.idx_opt = idx_opt;
  f.value = value;
  _filter.push_back(f);

  auto itr = _unique_attr.find(idx_col);
  if (itr == _unique_attr.end()) {
    _unique_attr.insert(pair<int, int>(idx_col, 1));
  } else {
    itr->second += 1;
  }

  stat_tabel();
}

void Tdataframe::ReFilter()
{
  stat_tabel();
}

void Tdataframe::clear_map_col_split()
{
  _map_col_split.clear();
}

void Tdataframe::add_filter(field_filter filter)
{
  _filter.push_back(filter);

  auto itr = _unique_attr.find(filter.idx_col);
  if (itr == _unique_attr.end()) {
    _unique_attr.insert(pair<int, int>(filter.idx_col, 1));
  } else {
    itr->second += 1;
  }

  stat_tabel();
}


void Tdataframe::info()
{
  cout << " Info" << endl;
  cout << " Nama File   : " << _nm_file << endl;
  cout << " Jumlah Data : " << _jml_row << endl;

  cout << _stat_label << endl ;
}

void Tdataframe::split_data(int split_column, string split_value, Tdataframe &data_below, Tdataframe &data_above)
{
  _data.clear_memory();
  if (split_value != "-1") {
    if (_data_type[split_column] == "continuous.")
    {
      data_below.add_filter(split_column, 0, split_value);
      data_above.add_filter(split_column, 1, split_value);

    } else {
      data_below.add_filter(split_column, 2, split_value);
      data_above.add_filter(split_column, 3, split_value);
    }
  }
}

void Tdataframe::get_col_pot_split(int idx)
{
  std::lock_guard<std::mutex> lock(v_mutex);
  _data.reset_file();
  while (!_data.is_eof())
  {
    _map_col_split.add_data(idx, _data.get_col_val(idx), _data_type[idx], _data.get_col_val(_idx_label));
    _data.next_record();
  }

  _map_col_split.cek_valid_attr(_jml_row);
}


void Tdataframe::calculate_metric(map<Tmy_dttype, Tlabel_stat>* _col_pot_split, float & current_overall_metric, string & split_value, Tlabel_stat & stat_label)
{
  float entropy_before_split;

  if (!config->use_credal) {
    entropy_before_split = stat_label.get_entropy();
  } else {
    entropy_before_split = stat_label.get_credal_entropy();
  }

  float best_overall_metric = 0.0;
  float gain = 0, gain_max = 0;
  string mid_point = "0.0";
  string tmp_split_value = "-1";

  bool first_iteration = true;

  Tlabel_stat _stat_label_below;


  auto itr_next = _col_pot_split->begin();
  itr_next++;

  auto itr = _col_pot_split->begin();

  size_t i = 1;
  while ((itr != _col_pot_split->end()))
  {
    _stat_label_below = _stat_label_below + (*itr).second;
    _stat_label_below.set_config(config);

    if (((itr != itr_next) and (itr_next != _col_pot_split->end()) ))
    {
      Tmy_dttype tmp1 = (*itr).first;
      Tmy_dttype tmp2 = (*itr_next).first;
      try {
        mid_point = to_string((stof(tmp1.get_string()) + stof(tmp2.get_string())) / 2);
      }
      catch (const std::invalid_argument& ia) {
        cout << tmp1.get_string() << "+" << tmp2.get_string() << " ";

      }

      Tbelow_above ba(config);
      //ba.set_value(mid_point);

      ba.add_below(_stat_label_below);
      Tlabel_stat tmp_stat = stat_label - _stat_label_below;
      tmp_stat.set_config(config);
      ba.add_above(tmp_stat);

      gain = 0;
      if (ba.cek_valid()) {
        float entropy_after_split = ba.get_overall_metric();
        float split_info = ba.get_split_info();
        gain = (entropy_before_split - entropy_after_split) / split_info;
      }

    }

    itr_next++;
    itr++;
    i++;

    if ((first_iteration and (gain > 0)) or (gain_max < gain))
    {
      first_iteration = false;
      gain_max = gain;
      tmp_split_value = mid_point;
      best_overall_metric = gain_max;
    }

  }

  current_overall_metric = best_overall_metric;
  split_value = tmp_split_value;

}

void Tdataframe::handle_continuous(int idx, float & current_overall_metric, string & split_value)
{

  map<Tmy_dttype, Tlabel_stat>* _col_pot_split = _map_col_split.get_pot_split(idx);

  if (_col_pot_split->size() > 0)
  {
    if (_col_pot_split->size() == 1)
    {
      float entropy_before_split;

      if (!config->use_credal) {
        entropy_before_split = _stat_label.get_entropy();
      } else {
        entropy_before_split = _stat_label.get_credal_entropy();
      }

      //cetak("{ ==1 start ");
      auto itr = _col_pot_split->begin();


      Tlabel_stat _stat_label_below = (*itr).second;
      _stat_label_below.set_config(config);

      Tmy_dttype tmp1 = (*itr).first;

      Tbelow_above ba(config);
      ba.set_value(tmp1);
      ba.add_below(_stat_label_below);

      Tlabel_stat tmp_stat = _stat_label - _stat_label_below;
      tmp_stat.set_config(config);
      ba.add_above(tmp_stat);

      float gain = 0;
      if (ba.cek_valid()) {
        float entropy_after_split = ba.get_overall_metric();
        float split_info = ba.get_split_info();
        gain = (entropy_before_split - entropy_after_split) / split_info;// 0;
      }
      if (gain > 0) {
        current_overall_metric = gain;
        split_value = tmp1.get_string();
      }
      //cetak(" ==1 end }");

    } else {
      float tmp_best_overall_metric = 0.0;
      string tmp_split_value = "-1";
      calculate_metric(_col_pot_split, tmp_best_overall_metric, tmp_split_value, _stat_label);

      current_overall_metric = tmp_best_overall_metric;
      split_value = tmp_split_value;
    }

  }
}

void Tdataframe::handle_non_continuous(int idx, float & current_overall_metric, string & split_value)
{
  float entropy_before_split;

  if (!config->use_credal) {
    entropy_before_split = _stat_label.get_entropy();
  } else {
    entropy_before_split = _stat_label.get_credal_entropy();
  }

  float best_overall_metric = 0.0;
  float gain = 0, gain_max = 0;
  string tmp_split_value = "-1";
  string mid_point = "0.0";

  bool first_iteration = true;

  map<Tmy_dttype, Tlabel_stat>* _col_pot_split = _map_col_split.get_pot_split(idx);

  auto itr = _col_pot_split->begin();
  while (itr != _col_pot_split->end())
  {
    Tbelow_above ba(config);

    mid_point = ((Tmy_dttype) (*itr).first).get_string();

    //ba.set_value(mid_point);
    ba.add_below((*itr).second);
    Tlabel_stat tmp_stat = _stat_label - (*itr).second;
    tmp_stat.set_config(config);
    ba.add_above(tmp_stat);

    gain = 0;
    if (ba.cek_valid()) {
      float entropy_after_split = ba.get_overall_metric();
      float split_info = ba.get_split_info();
      gain = (entropy_before_split - entropy_after_split) / split_info;
    }

    itr++;

    if ((first_iteration and (gain > 0)) or (gain_max < gain))
    {
      first_iteration = false;
      gain_max = gain;
      tmp_split_value = mid_point;
      best_overall_metric = gain_max;
    }

  }

  current_overall_metric = best_overall_metric;
  split_value = tmp_split_value;
}

void Tdataframe::calculate_overall_metric(int idx, float & current_overall_metric, string & split_value)
{
  std::lock_guard<std::mutex> lock(v_mutex);
  split_value = "-1";
  current_overall_metric = -1;

  //if (_map_col_split.is_valid_attr(idx)) {

  if (_data_type[idx] == "continuous.") {
    handle_continuous(idx, current_overall_metric, split_value);
  } else {
    handle_non_continuous(idx, current_overall_metric, split_value);
  }
  //}

  //cetak(" split value %s ",split_value.c_str());

  //_map_col_split.clear();
}

string Tdataframe::get_nm_header(int idx_col)
{
  return _data_header[idx_col];
}

int Tdataframe::get_opt(int idx_col, int is_below)
{
  if (_data_type[idx_col] == "continuous.")
  {
    return (is_below == 1) ? 0 : 1;

  } else {
    return (is_below == 1) ? 2 : 3;
  }
}


float Tdataframe::get_entropy()
{
  return _stat_label.get_entropy();
}

vector<string> Tdataframe::get_list_label()
{
  return _list_label;
}