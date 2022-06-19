#include <functional>
#include "Tmy_double.h"
#include "T_alpha_container.h"
#include "Tmy_alpha.h"
#include "Tmy_kernel.h"

#ifndef Included_T_grad_container_H
#define Included_T_grad_container_H

class T_grad_container
{
private:
	vector<Tmy_double> _grad;
	vector<int> _idx;
	vector<bool> _is_kkt;

public:
	T_grad_container();
	~T_grad_container();

	void reserve(size_t n);
	void assign(size_t n, Tmy_double value);

	Tmy_double &operator[](size_t idx);
	Tmy_double obj(size_t idx, Tmy_double rho);
	Tmy_double dec(size_t idx, Tmy_double rho);

	void mv_idx(int idx, int flag);
	void set_kkt(int idx, bool val);
	bool get_kkt(int idx);
	vector<int> get_rand_idx();
};

#endif