#include <string>
#include <iostream>
using namespace std;


#ifndef Included_Tmy_dttype_H

#define Included_Tmy_dttype_H

class Tmy_dttype
{
private:
	string _value;
	string _lower_value;
	bool _is_continue;

	string to_lower(const string str);

public:
	Tmy_dttype();
	~Tmy_dttype();
	Tmy_dttype(string value, bool is_continue);


	void set_value(string value, bool is_continue);
	string get_value();

	string get_string();

	bool is_continue();


	Tmy_dttype& operator = (const Tmy_dttype &t)
	{
		this->_value = t._value;
		this->_lower_value = t._lower_value;
		this->_is_continue = t._is_continue;

		return *this;
	}

	bool operator <(const Tmy_dttype& rhs) const
	{
		if (_is_continue and rhs._is_continue)
		{
			try {
				return  (stof(_value) < stof(rhs._value));
			}
			catch (const std::invalid_argument& ia) {
				cout << _value << "<" << rhs._value << " ";
				return false;
			}

		} else {
			if (!_is_continue and !rhs._is_continue)
			{
				return (_lower_value.compare(rhs._lower_value) < 0);
			} else {
				return false;
			}
		}
	}

	bool operator <=(const Tmy_dttype& rhs) const
	{
		if (_is_continue and rhs._is_continue)
		{
			try {
				return  (stof(_value) <= stof(rhs._value));
			}
			catch (const std::invalid_argument& ia) {
				cout << _value << "<=" << rhs._value << " ";
				return false;
			}
		} else {
			if (!_is_continue and !rhs._is_continue)
			{
				return (_lower_value.compare(rhs._lower_value) <= 0);
			} else {
				return false;
			}
		}
	}

	bool operator <=(const string& rhs) const
	{
		if (_is_continue)
		{
			
			return  (stof(_value) <= stof(rhs));

		} else {
			if (!_is_continue)
			{
				return (_value.compare(rhs) <= 0);
			} else {
				return false;
			}
		}
	}

	bool operator >(const Tmy_dttype& rhs) const
	{
		if (_is_continue and rhs._is_continue)
		{
			return  (stof(_value) > stof(rhs._value));
		} else {
			if (!_is_continue and !rhs._is_continue)
			{
				return (_lower_value.compare(rhs._lower_value) > 0);
			} else {
				return false;
			}
		}
	}

	bool operator >(const string& rhs) const
	{
		if (_is_continue)
		{
			return  (stof(_value) > stof(rhs));
		} else {
			if (!_is_continue)
			{
				return (_value.compare(rhs) > 0);
			} else {
				return false;
			}
		}
	}

	bool operator >=(const Tmy_dttype& rhs) const
	{
		if (_is_continue and rhs._is_continue)
		{
			return  (stof(_value) >= stof(rhs._value));
		} else {
			if (!_is_continue and !rhs._is_continue)
			{
				return (_lower_value.compare(rhs._lower_value) >= 0);
			} else {
				return false;
			}
		}
	}

	bool operator ==(const Tmy_dttype& rhs) const
	{
		if (_is_continue and rhs._is_continue)
		{
			try {
				return  (stof(_value) == stof(rhs._value));
			}
			catch (const std::invalid_argument& ia) {
				cout << _value << "==" << rhs._value << " ";
				return false;
			}
		} else {
			if (!_is_continue and !rhs._is_continue)
			{
				return (_lower_value.compare(rhs._lower_value) == 0);
			} else {
				return false;
			}
		}
	}

	bool operator ==(const string& rhs) const
	{

		if (_is_continue)
		{
			return  (stof(_value) == stof(rhs));
		} else {
			if (!_is_continue)
			{
				return (_value.compare(rhs) == 0);
			} else {
				return false;
			}
		}
	}

	bool operator !=(const Tmy_dttype& rhs) const
	{

		if (_is_continue and rhs._is_continue)
		{
			try {
				return  (stof(_value) != stof(rhs._value));
			}
			catch (const std::invalid_argument& ia) {
				cout << _value << "!=" << rhs._value << " ";
				return false;
			}
		} else {
			if (!_is_continue and !rhs._is_continue)
			{
				return (_lower_value.compare(rhs._lower_value) != 0);
			} else {
				return false;
			}
		}
	}

	bool operator !=(const string& rhs) const
	{

		if (_is_continue)
		{
			return  (stof(_value) != stof(rhs));
		} else {
			if (!_is_continue)
			{
				return (_value.compare(rhs) != 0);
			} else {
				return false;
			}
		}
	}

};

#endif