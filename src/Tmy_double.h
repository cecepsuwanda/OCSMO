#include <string>
#include <iostream>
#include <cmath>
#include "global.h"
using namespace std;

#ifndef Included_Tmy_double_H

#define Included_Tmy_double_H

class Tmy_double
{

	friend ostream &operator<<(ostream &os, const Tmy_double &rhs)
	{
		os << rhs._val;
		return os;
	}

	friend Tmy_double operator*(const double &lhs, const Tmy_double &rhs)
	{
		double tmp = (lhs * rhs._val);
		return bulat_nol(tmp, 1e-10, 10);
	}

	friend Tmy_double operator/(const double &lhs, const Tmy_double &rhs)
	{
		double tmp = (lhs / rhs._val);
		return bulat_nol(tmp, 1e-10, 10);
	}

	friend Tmy_double operator-(const double &lhs, const Tmy_double &rhs)
	{
		double tmp = lhs;
		double tmp1 = rhs._val;
		tmp = tmp - tmp1;
		return bulat_nol(tmp, 1e-10, 10);
	}

	friend double operator+(const double &lhs, const Tmy_double &rhs)
	{
		double tmp = lhs;
		double tmp1 = rhs._val;
		tmp = tmp + tmp1;
		return bulat_nol(tmp, 1e-10, 10);
	}

	friend Tmy_double abs(const Tmy_double &rhs)
	{
		return abs(rhs._val);
	}

private:
	double _val;
	double _batas = 1e-10;

public:
	Tmy_double();
	~Tmy_double();

	Tmy_double(const Tmy_double &t)
	{
		_val = t._val;
	}

	Tmy_double(const double t)
	{
		_val = t;
	}

	Tmy_double &operator=(const Tmy_double &t)
	{
		this->_val = t._val;
		return *this;
	}

	Tmy_double &operator=(const double t)
	{
		this->_val = t;
		return *this;
	}

	Tmy_double operator*(const Tmy_double &rhs)
	{
		double tmp = _val * rhs._val;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator*(const double &rhs)
	{
		double tmp = _val * rhs;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator/(const Tmy_double &rhs)
	{
		double tmp = _val / rhs._val;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator/(const double &rhs)
	{
		double tmp = _val / rhs;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator+(const Tmy_double &rhs)
	{
		double tmp = _val + rhs._val;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator+(const double &rhs)
	{
		double tmp = _val + rhs;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator-(const Tmy_double &rhs)
	{
		double tmp = _val - rhs._val;
		return bulat_nol(tmp, 1e-10, 10);
	}

	Tmy_double operator-(const double &rhs)
	{
		double tmp = _val - rhs;
		return bulat_nol(tmp, 1e-10, 10);
	}

	bool operator!=(const Tmy_double &rhs) const
	{
		double tmp = rhs._val;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(abs(diff) > _batas)
		return (!stat);
	}

	bool operator!=(const double &rhs) const
	{
		double tmp = rhs;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(abs(diff) > _batas)
		return (!stat);
	}

	bool operator<(const Tmy_double &rhs) const
	{
		double tmp = rhs._val;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//((abs(diff) > _batas) and (tmp1 < tmp))
		return (!stat and tmp1 < tmp);
	}

	bool operator<(const double &rhs) const
	{
		double tmp = rhs;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//((abs(diff) > _batas) and (tmp1 < tmp))
		return (!stat and tmp1 < tmp);
	}

	bool operator>(const Tmy_double &rhs) const
	{
		double tmp = rhs._val;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//((abs(diff) > _batas) and (tmp1 > tmp))
		return (!stat and tmp1 > tmp);
	}

	bool operator>(const double &rhs) const
	{
		double tmp = rhs;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//((abs(diff) > _batas) and (tmp1 > tmp))
		return (!stat and tmp1 > tmp);
	}

	bool operator>=(const Tmy_double &rhs) const
	{
		double tmp = rhs._val;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(((abs(diff) > _batas) and (tmp1 > tmp)) or (abs(diff) < _batas))
		return ((!stat and tmp1 > tmp) or stat);
	}

	bool operator>=(const double &rhs) const
	{

		double tmp = rhs;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(((abs(diff) > _batas) and (tmp1 > tmp)) or (abs(diff) < _batas))
		return ((!stat and tmp1 > tmp) or stat);
	}

	bool operator<=(const Tmy_double &rhs) const
	{
		double tmp = rhs._val;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(((abs(diff) > _batas) and (tmp1 < tmp)) or (abs(diff) < _batas))
		return ((!stat and tmp1 < tmp) or stat);
	}

	bool operator<=(const double &rhs) const
	{
		double tmp = rhs;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(((abs(diff) > _batas) and (tmp1 < tmp)) or (abs(diff) < _batas))
		return ((!stat and tmp1 < tmp) or stat);
	}

	bool operator==(const Tmy_double &rhs) const
	{
		double tmp = rhs._val;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp1, tmp);
		// double diff = tmp1 - tmp;
		//(abs(diff) < _batas)
		return (stat);
	}

	bool operator==(const double &rhs) const
	{
		double tmp = rhs;
		double tmp1 = _val;
		bool stat = AlmostEqualRelative(tmp, tmp1);
		// double diff = tmp1 - tmp;
		//(abs(diff) < _batas)
		return (stat);
	}

	operator double()
	{
		return _val;
	}

	operator int()
	{
		return (int)_val;
	}
};

#endif