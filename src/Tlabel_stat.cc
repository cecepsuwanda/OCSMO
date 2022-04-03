#include "Tlabel_stat.h"


Tlabel_stat::Tlabel_stat()
{
	_jml_row = 0;
	_max_value = 0;
}

Tlabel_stat::Tlabel_stat(Tconfig* v_config)
{
	_jml_row = 0;
	_max_value = 0;

	config = v_config;
}

Tlabel_stat::~Tlabel_stat()
{
	clear();
}

void Tlabel_stat::set_config(Tconfig* v_config)
{
	config = v_config;
}


void Tlabel_stat::clear()
{
	_jml_row = 0;
	_map.clear();
}

int Tlabel_stat::get_jml_row()
{
	return _jml_row;
}

int Tlabel_stat::get_jml_row_in_map()
{
	return _map.size();
}

string Tlabel_stat::get_first_value_in_map()
{
	auto itr = _map.begin();
	string tmp_string = itr->first;
	return tmp_string;
}


double Tlabel_stat::get_entropy()
{
	double entropy = 0;
	credal crd(config->credal_s);

	vector<int> freq;

	std::transform(_map.begin(), _map.end(),
	               std::back_inserter(freq),
	[](const std::pair<string, int> &p) {
		return p.second;
	});


	crd.input_frec(freq);

	entropy = crd.get_ent();

	/*auto it = _map.begin();
	while (it != _map.end())
	{
		if ( (it->second > 0) and (_jml_row > 0) ) {
			entropy += ((float) it->second / _jml_row)  * (-1 * log2((float) it->second / _jml_row));
		}

		it++;
	}*/

	return entropy;
}

double Tlabel_stat::get_credal_entropy()
{
	double entropy = 0;
	credal crd(config->credal_s);

	vector<int> freq;

	std::transform(_map.begin(), _map.end(),
	               std::back_inserter(freq),
	[](const std::pair<string, int> &p) {
		return p.second;
	});


	crd.input_frec(freq);

	entropy = crd.get_max_ent();

	return entropy;
}

float Tlabel_stat::get_estimate_error()
{
	float estimate_error = 0;

	float f = (float) _min_value / _jml_row;
	float z = 0.69;

	estimate_error = (f + ((z * z) / (2 * _jml_row)) + (z * sqrt((f / _jml_row) - ((f * f) / _jml_row) + ((z * z) / (4 * _jml_row * _jml_row)))) ) / (1 + ((z * z) / _jml_row));

	return estimate_error;
}

void Tlabel_stat::add(string value)
{
	_jml_row += 1;
	map<string, int>::iterator it;

	it = _map.find(value);
	if (it == _map.end())
	{
		_map.insert(pair<string, int>(value, 1));
	} else {
		it->second += 1;
	}

	auto itr = _map.begin();
	_max_label = itr->first;
	_max_value = itr->second;

	while (itr != _map.end())
	{
		if (itr != _map.begin())
		{
			if (_max_value < itr->second)
			{
				_max_label = itr->first;
				_max_value = itr->second;
			}
		}

		itr++;
	}

	_min_value = _jml_row - _max_value;

}

bool Tlabel_stat::is_single_label()
{
	return _map.size() == 1;
}

string Tlabel_stat::get_max_label()
{
	return _max_label;
}

map<string, int> Tlabel_stat::get_map()
{
	return _map;
}

int Tlabel_stat::get_value(string key)
{
	return _map[key];
}

ostream & operator << (ostream &out, const Tlabel_stat &tc)
{
	if (tc._map.size() > 0)
	{
		out << " Statistik   : " << endl;
		out << std::setw(30) << "label" << std::setw(10) << "Jumlah" << endl;
		for (auto it = tc._map.begin(); it != tc._map.end(); ++it) {
			out << std::setw(30) << (*it).first << std::setw(10) << (*it).second << endl;
		}
	}
	return out;
}