#include "Tmy_alpha.h"

Tmy_alpha::Tmy_alpha(Tconfig *v_config)
{
	_config = v_config;
}

Tmy_alpha::~Tmy_alpha()
{
}

void Tmy_alpha::init(int jml_data, T_alpha_container &alpha)
{
	Tmy_double ub = _config->eps / (_config->V * jml_data);

	alpha.boundaries(0.0, ub);
	alpha.reserve(jml_data);
	alpha.assign(jml_data, 0.0);

	Tmy_double tmp = _config->V * ((double)jml_data);
	int jml = (int)tmp;

	for (int i = 0; i < jml; ++i)
	{
		alpha[i] = ub;
	}

	Tmy_double jml_alpha = alpha.sum();

	if (jml_alpha < _config->eps)
	{
		alpha[jml] = _config->eps - jml_alpha;
	}

	int jml_sv = alpha.n_sv();

	if (jml_sv == 0)
	{
		alpha[0] = ub / 4.0;
		alpha[1] = alpha[1] + (ub - (ub / 4.0));
	}
}

vector<Tmy_double> Tmy_alpha::calculateBoundaries(int i, int j, T_alpha_container alpha)
{
	Tmy_double t = alpha[i] + alpha[j];
	Tmy_double diff = 0.0;
	Tmy_double diff1 = 0.0;
	diff = t - alpha.ub();
	diff1 = t + alpha.lb();
	vector<Tmy_double> hasil = {alpha.lb(), alpha.ub()};
	if (((alpha[i] <= alpha.ub()) and (alpha[i] >= alpha.lb())) and ((alpha[j] <= alpha.ub()) and (alpha[j] >= alpha.lb())))
	{
		hasil = {max(diff, alpha.lb()), min(alpha.ub(), diff1)};
	}
	return hasil;
}

vector<Tmy_double> Tmy_alpha::limit_alpha(Tmy_double alpha_a, Tmy_double alpha_b, Tmy_double Low, Tmy_double High, int flag)
{
	vector<Tmy_double> hasil = {alpha_a, alpha_b};
	if (alpha_a > High)
	{
		if (flag == 1)
		{
			Tmy_double s = alpha_a - High;
			hasil[1] = alpha_b + s;
		}
		hasil[0] = High;
	}
	else
	{
		if (alpha_a < Low)
		{
			if (flag == 1)
			{
				Tmy_double s = alpha_a - Low;
				hasil[1] = alpha_b + s;
			}
			hasil[0] = Low;
		}
	}
	return hasil;
}

vector<Tmy_double> Tmy_alpha::calculateNewAlpha(int i, int j, Tmy_double delta, Tmy_double Low, Tmy_double High, T_alpha_container alpha)
{
	Tmy_double alpha_a_new = alpha[i] + delta;
	vector<Tmy_double> tmp = limit_alpha(alpha_a_new, 0, Low, High, 0);
	alpha_a_new = tmp[0];
	Tmy_double alpha_b_new = alpha[j] + (alpha[i] - alpha_a_new);
	tmp = limit_alpha(alpha_b_new, alpha_a_new, alpha.lb(), alpha.ub(), 1);
	alpha_b_new = tmp[0];
	alpha_a_new = tmp[1];
	return {alpha[i], alpha[j], alpha_a_new, alpha_b_new};
}

Treturn_is_pass Tmy_alpha::is_pass(int i, int j, Tmy_double delta, T_alpha_container alpha)
{
	Treturn_is_pass tmp;
	tmp.is_pass = false;
	tmp.alpha_i = alpha[i];
	tmp.alpha_j = alpha[j];
	tmp.new_alpha_i = alpha[i];
	tmp.new_alpha_j = alpha[j];
	tmp.lb = alpha.lb();
	tmp.ub = alpha.ub();

	if (i == j)
	{
		return tmp;
	}
	else
	{
		vector<Tmy_double> hsl = calculateBoundaries(i, j, alpha);
		Tmy_double Low = hsl[0], High = hsl[1];
		// cout <<"Low "<<Low<<" High "<<High<<endl;
		if (Low == High)
		{
			return tmp;
		}
		else
		{
			vector<Tmy_double> hsl = calculateNewAlpha(i, j, delta, Low, High, alpha);
			Tmy_double alpha_a_old = hsl[0], alpha_b_old = hsl[1], alpha_a_new = hsl[2], alpha_b_new = hsl[3];
			double diff = alpha_a_new - alpha_a_old;
			// abs(diff)<10e-5
			if (abs(diff) < 1e-5)
			{
				return tmp;
			}
			else
			{
				tmp.is_pass = true;
				tmp.alpha_i = alpha_a_old;
				tmp.alpha_j = alpha_b_old;
				tmp.new_alpha_i = alpha_a_new;
				tmp.new_alpha_j = alpha_b_new;
				// cout<<"alpha_a_new : "<<alpha_a_new<<" alpha_a_old : "<<alpha_a_old<<endl;
				// cout<<"alpha_b_new : "<<alpha_b_new<<" alpha_b_old : "<<alpha_b_old<<endl;
				return tmp;
			}
		}
	}

	return tmp;
}
