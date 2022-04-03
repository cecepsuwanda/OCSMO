
#include <map>
#include <cmath>
#include <iterator>
#include <string>
#include <iomanip>
#include <algorithm>
#include "credal.h"
#include "global.h"

using namespace std;

#ifndef Included_Tlabel_stat_H

#define Included_Tlabel_stat_H

class Tlabel_stat
{
private:
	int _jml_row;
	map<string, int> _map;
	string _max_label;
	int _max_value;
	int _min_value;
	Tconfig* config;
public:
	Tlabel_stat();
	Tlabel_stat(Tconfig* v_config);
	~Tlabel_stat();
	
	void set_config(Tconfig* v_config);
	
	void add(string value);
	void clear();
    int get_jml_row();

    double get_entropy();
    double get_credal_entropy();
    
    float get_estimate_error();
    map<string, int> get_map();
    int get_value(string key);

    int get_jml_row_in_map();
    string get_first_value_in_map();

    string get_max_label();
    bool is_single_label();

	Tlabel_stat& operator = (const Tlabel_stat &t)
	{
		this->_jml_row = t._jml_row;

		this->_map.clear();

		for (auto it = t._map.begin(); it != t._map.end(); it++)
		{
			this->_map.insert(pair<string, int>(it->first, it->second));
		}

		this->_max_value = t._max_value;
		this->_max_label = t._max_label;
		this->_min_value = t._min_value;
		this->config = t.config;

		return *this;
	}

	const Tlabel_stat operator + (const Tlabel_stat &t) const
	{
		Tlabel_stat tmp;

		tmp._jml_row =  this->_jml_row + t._jml_row;

		tmp._map.clear();

		for (auto it = t._map.begin(); it != t._map.end(); it++)
		{
			auto it1 = this->_map.find(it->first);
			if (it1 == this->_map.end())
			{
				tmp._map.insert(pair<string, int>(it->first, it->second));
			} else {
				tmp._map.insert(pair<string, int>(it->first, it->second + it1->second));
			}
		}

		for (auto it = this->_map.begin(); it != this->_map.end(); it++)
		{
			auto it1 = tmp._map.find(it->first);
			if (it1 == tmp._map.end())
			{
				tmp._map.insert(pair<string, int>(it->first, it->second));
			}
		}

		return tmp;
	}

	const Tlabel_stat operator - (const Tlabel_stat &t) const
	{
		Tlabel_stat tmp;

		tmp._jml_row =  this->_jml_row - t._jml_row;

		tmp._map.clear();

        
        for (auto it = this->_map.begin(); it != this->_map.end(); it++)
		{
			auto it1 = t._map.find(it->first);
			if (it1 == t._map.end())
			{
				tmp._map.insert(pair<string, int>(it->first, it->second));
			} else {
				tmp._map.insert(pair<string, int>(it->first, it->second - it1->second));
			}
		}

		for (auto it = t._map.begin(); it != t._map.end(); it++)
		{
			auto it1 = tmp._map.find(it->first);
			if (it1 == tmp._map.end())
			{
				tmp._map.insert(pair<string, int>(it->first, -it->second));
			}
		}		  

		

		return tmp;
	}


    friend ostream & operator << (ostream & out, const Tlabel_stat & tc);

};

#endif