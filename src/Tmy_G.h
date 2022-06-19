#include <string>
#include <cstdlib>
#include <ctime>
#include "Tmy_kernel.h"
#include "T_alpha_container.h"
#include "T_grad_container.h"
#include "Tmy_alpha.h"
#include "Tmy_double.h"

using namespace std;

#ifndef Included_Tmy_G_H

#define Included_Tmy_G_H

struct callback_param
{
	int idx = -1;
	Tmy_double grad = 0.0;
	Tmy_double obj = 0.0;
	Tmy_double dec = 0.0;
};

using namespace std::placeholders;
using callback_type = std::function<bool(callback_param, callback_param, T_alpha_container, T_grad_container, Tmy_kernel *)>;
using callback_type1 = std::function<bool(callback_param, callback_param, T_alpha_container, T_grad_container, Tmy_kernel *, Tmy_alpha *)>;

class Tmy_G
{
private:
	int _jml_data;

	int max(Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, callback_type f);
	int max(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, callback_type f);
	int max(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, Tmy_alpha *my_alpha, callback_type1 f);
	int cari(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, Tmy_alpha *my_alpha, callback_type1 f);

	bool delta_filter(int idx_b, int idx_a, T_alpha_container alpha, Tmy_double delta);

public:
	Tmy_G();
	~Tmy_G();

	void init(int jml_data, Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container &grad);
	Tmy_double update_rho(Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container grad);
	Tmy_double update_rho(int idx_b, int idx_a, Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container grad);

	bool is_kkt(int idx, Tmy_double rho, T_alpha_container alpha, T_grad_container grad);

	void set_kkt(Tmy_double rho, T_alpha_container alpha, T_grad_container &grad);

	int cari_idx_a(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel);

	int cari_idx_lain(int idx_b, Tmy_double rho, Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container grad, Tmy_alpha *my_alpha);

	bool cari_idx(int &idx_b, int &idx_a, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel);

	void update_G(int idx_b, int idx_a, Treturn_is_pass tmp, Tmy_kernel *kernel, T_alpha_container &alpha, T_grad_container &grad);
};

#endif