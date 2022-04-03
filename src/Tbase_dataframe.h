#include "Tread_file.h"
#include "Tlabel_stat.h"
#include "Tmy_dttype.h"
#include "Tmap_col_split.h"
#include "global.h"
#include <string>

using namespace std;

#ifndef Included_Tbase_dataframe_H

#define Included_Tbase_dataframe_H

struct field_filter
{
	int idx_col;
	int idx_opt;
	string value;
};

class Tbase_dataframe
{
protected:

	Tread_file _data;
	vector<string> _data_header;
	vector<string> _data_type;
	vector<field_filter> _filter;
	int _jml_col = 0;
	int _jml_row = 0;
	int _jml_total_row = 0;
	string _nm_file;
	int _id = 0;

	bool _by_pass_filter = false;

	mutable std::mutex v_mutex;

private:

public:
	Tbase_dataframe();
	~Tbase_dataframe();

	Tbase_dataframe(const Tbase_dataframe &t)
	{
		_data = t._data;
		_data_header = t._data_header;
		_data_type = t._data_type;
		_filter = t._filter;		

		_jml_col = t._jml_col;
		_jml_row = t._jml_row;		

		_nm_file = t._nm_file;		
		_jml_total_row = t._jml_total_row;
	}


	Tbase_dataframe& operator = (const Tbase_dataframe &t)
	{
		this->_data = t._data;
		this->_data_header = t._data_header;
		this->_data_type = t._data_type;
		this->_filter = t._filter;		
		this->_jml_col = t._jml_col;
		this->_jml_row = t._jml_row;		
		this->_nm_file = t._nm_file;		
		this->_jml_total_row = t._jml_total_row;
		return *this;
	}

	void set_id(int id);
	int get_id();

	void by_pass_filter_on();
	void by_pass_filter_off();	

	void read_data(string nm_f);
	void read_data_type(string nm_f);
	void save_to(string nm_file);
	string get_data_type(int idx);

	bool is_pass(vector<string> &data);
	bool is_pass();

	void stat_tabel();
	
	int getjmlcol();
	int getjmlrow();
	void setjmltotalrow();
	int getjmltotalrow();

	void reset_file();
	void close_file();

	bool is_eof();
	void next_record();
	vector<string> get_record();
	vector<vector<string>> get_all_record();

	void next_col();
	bool is_end_col();
	string get_col_val();
	string get_col_val(int idx_col);
	int get_idx_col();

	void add_filter(int idx_col, int idx_opt, string value);
	void add_filter(field_filter filter);
	void ReFilter();
	vector<field_filter> get_filter();

	void clear_memory();

	void info();
	void head();

};

#endif