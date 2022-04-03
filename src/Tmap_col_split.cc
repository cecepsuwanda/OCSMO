#include "Tmap_col_split.h"


Tmap_col_split::Tmap_col_split()
{

}

Tmap_col_split::Tmap_col_split(Tconfig* v_config)
{
  config = v_config;
}

Tmap_col_split::~Tmap_col_split()
{

}

void Tmap_col_split::set_config(Tconfig* v_config)
{
	config = v_config;
}

void Tmap_col_split::add_data(int idx_col, string split_value, string tipe_data, string label)
{

	Tmy_dttype tmp_split_value(split_value, tipe_data == "continuous.");
	auto itr = _pot_split.find(idx_col);
	if (itr == _pot_split.end())
	{
		Tlabel_stat p(config);
		p.add(label);
		map<Tmy_dttype, Tlabel_stat> col_pot_split;
		col_pot_split.insert(pair<Tmy_dttype, Tlabel_stat>(tmp_split_value, p));
		_pot_split.insert(pair<int, map<Tmy_dttype, Tlabel_stat>>(idx_col, col_pot_split));

	} else {
		auto itr1 = itr->second.find(tmp_split_value);
		if (itr1 == itr->second.end())
		{
			Tlabel_stat p(config);
			p.add(label);
			itr->second.insert(pair<Tmy_dttype, Tlabel_stat>(tmp_split_value, p));
		} else {
			itr1->second.add(label);
		}
	}

}

void Tmap_col_split::cek_valid_attr(int jml_row)
{
	_valid_attr.clear();
	for (auto itr = _pot_split.begin(); itr != _pot_split.end(); ++itr)
	{

		auto tmp = itr->second.begin();
		Tmy_dttype tmp_dttype= tmp->first;

		if (tmp_dttype.is_continue()) {
			//cout << itr->first << "-" << itr->second.size() << endl;
			if ( itr->second.size() < (0.3 * jml_row) ) {				
				_valid_attr.push_back(itr->first);
			}
		} else {
			_valid_attr.push_back(itr->first);
		}
	}
}

int Tmap_col_split::get_jml_valid_attr()
{
	return _valid_attr.size();
}

int Tmap_col_split::get_valid_attr(int idx)
{
	return _valid_attr[idx];
}

bool Tmap_col_split::is_valid_attr(int idx)
{
	bool is_valid = false;
    if(_valid_attr.size()>0)
    { 
	    auto itr = _valid_attr.begin();
	    while((!is_valid) and (itr!=_valid_attr.end()))
	    {
	        is_valid = (*itr == idx);        
	        itr++;
	    }
    }
	return is_valid;
}

void Tmap_col_split::clear()
{
	for (auto i = _pot_split.begin(); i != _pot_split.end(); ++i)
	{
		for (auto j = i->second.begin(); j != i->second.end(); ++j)
		{
			j->second.clear();
		}
		i->second.clear();
	}
	_pot_split.clear();
}

map<Tmy_dttype, Tlabel_stat>* Tmap_col_split::get_pot_split(int idx)
{
	return &_pot_split[idx];
}