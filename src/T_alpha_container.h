#include <functional>
#include "Tmy_double.h"

#ifndef Included_T_alpha_container_H
#define Included_T_alpha_container_H

using sum_if_callback = std::function<bool(Tmy_double)>;

class T_alpha_container
{
private:
	vector<Tmy_double> _alpha;
	Tmy_double _lb;
	Tmy_double _ub;

public:
	T_alpha_container();
	~T_alpha_container();

	void reserve(size_t n);
	void assign(size_t n, Tmy_double value);
	void boundaries(Tmy_double lb, Tmy_double ub);
	Tmy_double sum();
	Tmy_double sum_if(sum_if_callback f);

	int n_all_sv();
	int n_sv();

	bool is_sv(size_t idx);
	bool is_ub(size_t idx);
	bool is_lb(size_t idx);
	bool is_nol(size_t idx);

	Tmy_double ub();
	Tmy_double lb();

	void nol_kan();

	Tmy_double &operator[](size_t idx);
};

#endif