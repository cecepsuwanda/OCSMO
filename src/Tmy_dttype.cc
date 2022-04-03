#include "Tmy_dttype.h"


Tmy_dttype::Tmy_dttype()
{
	_is_continue = false;
	_value = "-1";
}

Tmy_dttype::~Tmy_dttype()
{

}

Tmy_dttype::Tmy_dttype(string value, bool is_continue)
{
	_value = value;
	_lower_value = value;
	_is_continue = is_continue;
	if (!_is_continue)
	{
		to_lower(_lower_value);
	}
}

void Tmy_dttype::set_value(string value, bool is_continue)
{
	_value = value;
	_lower_value = value;
	_is_continue = is_continue;
	if (!_is_continue)
	{
		to_lower(_lower_value);
	}
}

string Tmy_dttype::get_value()
{
	return _value;
}

string Tmy_dttype::to_lower(const string str)
{
	string tmp_str;
	for (size_t i = 0; i < str.length(); ++i)
	{
		tmp_str[i] = toupper(str[i]);
	}
	return tmp_str;
}

string Tmy_dttype::get_string()
{
	return _value;
}

bool Tmy_dttype::is_continue()
{
	return _is_continue;
}