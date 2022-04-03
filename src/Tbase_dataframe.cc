#include "Tbase_dataframe.h"


Tbase_dataframe::Tbase_dataframe()
{

}

Tbase_dataframe::~Tbase_dataframe()
{
	_data_header.clear();
	_data_header.shrink_to_fit();
	_data_type.clear();
	_data_type.shrink_to_fit();
	_filter.clear();
	_filter.shrink_to_fit();
}

void Tbase_dataframe::set_id(int id)
{
	// cetak("{cid:");
	// cetak(to_string(id).c_str());
	// cetak("} ");
	_id = id;
}



int Tbase_dataframe::get_id()
{
	return _id;
}

void Tbase_dataframe::by_pass_filter_on(){
	_by_pass_filter = true;
}

void Tbase_dataframe::by_pass_filter_off(){
   _by_pass_filter = false;
}	


void Tbase_dataframe::read_data(string nm_f)
{
	//cout << "Tbase_dataframe read_data " << endl;
	
	_nm_file = nm_f;
	_data.setnm_f(nm_f, ",");
}


void Tbase_dataframe::read_data_type(string nm_f)
{
	//cout << "Tbase_dataframe read_data_type " << endl;
		
	Tread_file tmp;
	tmp.setnm_f(nm_f, ": ");

	int i = 0;
	while (!tmp.is_eof())
	{
		_data_header.push_back(tmp.get_col_val(0));
		_data_type.push_back(tmp.get_col_val(1));
		tmp.next_record();
		i++;
	}
	_jml_col = i;
	tmp.close_file();
	stat_tabel();
}


bool Tbase_dataframe::is_pass(vector<string> &data)
{
    
	bool pass = true;
	if (_filter.size() > 0)
	{
		size_t i = 0;
		while ((i < _filter.size()) and pass)
		{
			Tmy_dttype tmp1(_filter[i].value, _data_type[_filter[i].idx_col] == "continuous.");
			Tmy_dttype tmp2(data[_filter[i].idx_col], _data_type[_filter[i].idx_col] == "continuous.");

			switch (_filter[i].idx_opt)
			{
			case 0 :
				pass = tmp2 <= tmp1 ;
				break;
			case 1 :
				pass = tmp1  < tmp2;
				break;
			case 2 :
				pass = tmp1 == tmp2;
				break;
			case 3 :
				pass = tmp1 != tmp2;
				break;
			}
			i++;
		}
	}

	return pass;
}

bool Tbase_dataframe::is_pass()
{
    
	bool pass = true;
	if (_filter.size() > 0)
	{
		size_t i = 0;
		while ((i < _filter.size()) and pass)
		{
			Tmy_dttype tmp1(_filter[i].value, _data_type[_filter[i].idx_col] == "continuous.");
			Tmy_dttype tmp2(_data.get_col_val(_filter[i].idx_col), _data_type[_filter[i].idx_col] == "continuous.");

			switch (_filter[i].idx_opt)
			{
			case 0 :
				pass = tmp2 <= tmp1 ;
				break;
			case 1 :
				pass = tmp1  < tmp2;
				break;
			case 2 :
				pass = tmp1 == tmp2;
				break;
			case 3 :
				pass = tmp1 != tmp2;
				break;
			}
			i++;
		}
	}

	return pass;
}

void Tbase_dataframe::stat_tabel()
{
	_data.index_on();
	if ((_filter.size() > 0) or (_by_pass_filter==true)) {
		_data.clear_index();
	} else {
		_data.index_off();
	}

	int i = 0;

	_data.reset_file();
	while (!_data.is_eof())
	{
		
		if (is_pass())
		{
			if ((_filter.size() > 0) or (_by_pass_filter==true)) {
				_data.add_index();
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
}

int Tbase_dataframe::getjmlcol()
{
	return _jml_col;
}

int Tbase_dataframe::getjmlrow()
{
	return _jml_row;
}

void Tbase_dataframe::setjmltotalrow()
{
	_jml_total_row = _jml_row;
}

int Tbase_dataframe::getjmltotalrow()
{
	return _jml_total_row;
}

void Tbase_dataframe::reset_file()
{
	_data.reset_file();
}



void Tbase_dataframe::close_file()
{
	_data.close_file();
}

bool Tbase_dataframe::is_eof()
{
	return _data.is_eof();
}

void Tbase_dataframe::next_record()
{
	_data.next_record();
}

vector<string> Tbase_dataframe::get_record()
{
	return _data.get_record();
}

void Tbase_dataframe::next_col()
{
	_data.next_col();
}

bool Tbase_dataframe::is_end_col()
{
	return _data.is_end_col();
}

string Tbase_dataframe::get_col_val()
{
	return _data.get_col_val();
}

string Tbase_dataframe::get_col_val(int idx_col)
{
	return _data.get_col_val(idx_col);
}

int Tbase_dataframe::get_idx_col()
{
	return _data.get_idx_col();
}




vector<vector<string>> Tbase_dataframe::get_all_record()
{
	//std::lock_guard<std::mutex> lock(v_mutex);
    //ReFilter();

	vector<vector<string>> Table;

	_data.reset_file();
	while (!_data.is_eof())
	{
		Table.push_back(_data.get_record());
		_data.next_record();
	}

	//clear_memory();

	return Table;
}



void Tbase_dataframe::add_filter(int idx_col, int idx_opt, string value)
{
	
	field_filter f;
	f.idx_col = idx_col;
	f.idx_opt = idx_opt;
	f.value = value;
	_filter.push_back(f);

	stat_tabel();
}

void Tbase_dataframe::add_filter(field_filter filter)
{
	
	_filter.push_back(filter);
	stat_tabel();
}

void Tbase_dataframe::ReFilter()
{
	stat_tabel();
}

vector<field_filter> Tbase_dataframe::get_filter()
{
	return _filter;
}

void Tbase_dataframe::info()
{
	cout << " Info" << endl;
	cout << " Nama File   : " << _nm_file << endl;
	cout << " Jumlah Data : " << _jml_row << endl;
}

void Tbase_dataframe::save_to(string nm_file)
{
	// Tread_file tmp;
	// tmp.setnm_f(nm_file, ",");
	// tmp.open_file("w+");

	// vector<string> tmp_data;
	// _data.reset_file();
	// while (!_data.is_eof())
	// {
	// 	tmp_data = _data.get_record();

	// 	string tmp_str = "";
	// 	for (size_t i = 0; i < (tmp_data.size() - 1); ++i)
	// 	{
	// 		tmp_str = tmp_str + tmp_data[i] + ",";
	// 	}
	// 	tmp_str = tmp_str + tmp_data[tmp_data.size() - 1];

	// 	tmp.write_file(tmp_str);


	// 	tmp_data.clear();
	// 	tmp_data.shrink_to_fit();
	// 	_data.next_record();
	// }
}

string Tbase_dataframe::get_data_type(int idx)
{
	return _data_type[idx];
}

void Tbase_dataframe::head()
{
	vector<string> tmp_data;
	map<string, int>::iterator it;

	cout << " Head" << endl;
	cout << " Nama File   : " << _nm_file << endl;

	if (_jml_row > 0) {
		int i = 0;

		_data.reset_file();
		while ((!_data.is_eof()) and (i < 5))
		{
			tmp_data = _data.get_record();
			for (int idx = 0; idx < tmp_data.size(); ++idx)
			{
				cout << setw(tmp_data[idx].length() + 2) << tmp_data[idx];
			}
			cout << endl;
			i++;

			tmp_data.clear();
			tmp_data.shrink_to_fit();
			_data.next_record();
		}

	}
}

void Tbase_dataframe::clear_memory()
{
	_data.clear_memory();
}

