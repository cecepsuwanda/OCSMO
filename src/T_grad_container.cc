#include "T_grad_container.h"

T_grad_container::T_grad_container()
{
}

T_grad_container::~T_grad_container()
{
	// cout << " destroy grad container " << endl;
	_grad.clear();
	_idx.clear();
	_is_kkt.clear();
}

void T_grad_container::reserve(size_t n)
{
	_grad.reserve(n);
	_idx.reserve(n);
	_is_kkt.reserve(n);
}

void T_grad_container::assign(size_t n, Tmy_double value)
{
	_grad.assign(n, value);
	_is_kkt.assign(n, false);
	_idx.assign(n, 0);
	for (size_t i = 0; i < n; i++)
	{
		_idx[i] = i;
	}
}

Tmy_double &T_grad_container::operator[](size_t idx)
{
	return _grad.at(idx);
}

Tmy_double T_grad_container::obj(size_t idx, Tmy_double rho)
{
	return (_grad.at(idx) - rho);
}

Tmy_double T_grad_container::dec(size_t idx, Tmy_double rho)
{
	return (_grad.at(idx) - rho);
}

void T_grad_container::mv_idx(int idx, int flag)
{
	int i = 0;
	bool ketemu = false;
	while (!ketemu and (i < _idx.size()))
	{
		ketemu = _idx[i] == idx;
		i++;
	}
	if (ketemu)
	{
		_idx.erase(_idx.begin() + (i - 1));
		if (flag == 0)
		{
			_idx.insert(_idx.begin(), idx);
		}
		else
		{
			_idx.push_back(idx);
		}
	}
}

void T_grad_container::set_kkt(int idx, bool val)
{
	_is_kkt[idx] = val;
}

bool T_grad_container::get_kkt(int idx)
{
	return _is_kkt[idx];
}

vector<int> T_grad_container::get_rand_idx()
{
	return _idx;
}