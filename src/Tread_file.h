#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/shm.h>	 /* shared memory functions and structs. */
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <bits/stdc++.h>
#include <mutex>

using namespace std;

#ifndef Included_Tread_file_H

#define Included_Tread_file_H

#define pagesize  4096

class Tread_file {
private:
	string _nm_f;
	const char *_separator;
	vector<string> _data;
	string _data_str;

	int  _posisi = 0;
	int  _b_posisi = 0;
	int _fd = -1;
	char * _file_in_memory = NULL;
	struct stat _sb;

	int _idx_posisi = 0;
	bool is_index = false;
	vector<int> _index;

	int *_idx_in_memory = NULL;
	int _jml_index = 0;
	int _ukuran_index = 0;

	vector<string> tokenizer(char* str, const char* separator);
	void clear_data();
	bool open_file();

	int _idx_col = 0;

	int _jml_row = 0;
	int _jml_col = 0;

	bool _shared_memory = false;

	mutable std::mutex v_mutex;

	void memory_map_file();

public:
	Tread_file();
	~Tread_file();
	Tread_file(const Tread_file &t);
	Tread_file& operator = (const Tread_file &t)
	{
		this->_nm_f = t._nm_f;
		this->_separator = t._separator;

		this->_shared_memory = t._shared_memory;
		this->_idx_posisi = 0;
		this->_posisi = 0;

		this->_fd = t._fd;
		this->_sb = t._sb;

		if (_shared_memory)
		{
			memory_map_file();
		} else {
			this->_file_in_memory = t._file_in_memory;
		}


		if (t._jml_index > 0) {
			clear_index();
			for (int i = 0; i < t._jml_index; ++i)
			{
				this->_index.push_back(t._idx_in_memory[i]);
			}
			save_to_memory();
			clear_index();
		}

		return *this;
	}

	void set_shared_memory_on();
	void set_shared_memory_off();

	void setnm_f(string nm_f, const char* separator);
	void setseparator(const char* separator);
	int get_jml_row();
	int get_jml_col();


	bool open_file(string mode);
	void reset_file();
	void read_file();
	void close_file();

	bool is_eof();
	void next_record();
	vector<string> get_record();


	void next_col();
	bool is_end_col();
	string get_col_val();
	string get_col_val(int idx_col);
	int get_idx_col();

	void clear_index();
	void index_on();
	void index_off();
	void add_index();
	void save_to_memory();
	void clear_memory();

	void goto_rec(int idx);

};

#endif