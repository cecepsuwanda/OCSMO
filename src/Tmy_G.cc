#include "Tmy_G.h"

Tmy_G::Tmy_G()
{
}

Tmy_G::~Tmy_G()
{
}

void Tmy_G::init(int jml_data, Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container &grad)
{
  _jml_data = jml_data;

  grad.reserve(jml_data);
  grad.assign(jml_data, 0.0);

  for (int i = 0; i < _jml_data; ++i)
  {
    if (alpha[i] != 0.0)
    {
      vector<Tmy_double> data = kernel->get_Q(i);
      for (int j = 0; j < _jml_data; ++j)
      {
        grad[j] = grad[j] + (alpha[i] * data[j]);
      }
    }
  }
}

Tmy_double Tmy_G::update_rho(Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container grad)
{
  Tmy_double jml_G = 0.0;
  Tmy_double G_max = -HUGE_VAL;
  Tmy_double G_min = HUGE_VAL;

  int jml_sv = 0;
  for (int i = 0; i < _jml_data; ++i)
  {
    if (alpha.is_sv(i))
    {
      jml_G = jml_G + grad[i];
      jml_sv = jml_sv + 1;
    }
    else
    {
      if (alpha.is_lb(i))
      {
        if (grad[i] < G_min)
        {
          G_min = grad[i];
        }
      }
      else
      {
        if (alpha.is_ub(i))
        {
          if (grad[i] > G_max)
          {
            G_max = grad[i];
          }
        }
      }
    }
  }

  Tmy_double tmp_rho = 0.0;
  if (jml_sv > 0)
  {
    tmp_rho = jml_G / (1.0 * jml_sv);
  }
  else
  {
    tmp_rho = (G_min + G_max) / 2.0;
  }

  return tmp_rho;
}

Tmy_double Tmy_G::update_rho(int idx_b, int idx_a, Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container grad)
{
  Tmy_double tmp_rho = 0.0;

  if (alpha.is_sv(idx_b))
  {
    tmp_rho = grad[idx_b];
  }
  else
  {
    if (alpha.is_sv(idx_a))
    {
      tmp_rho = grad[idx_a];
    }
    else
    {
      tmp_rho = (grad[idx_b] + grad[idx_a]) / 2.0;
    }
  }

  return tmp_rho;
}

bool Tmy_G::is_kkt(int idx, Tmy_double rho, T_alpha_container alpha, T_grad_container grad)
{
  Tmy_double dec = grad.dec(idx, rho);
  bool stat1 = alpha.is_nol(idx) and (dec > 1e-3);
  bool stat2 = alpha.is_sv(idx) and (abs(dec) <= 1e-3);
  bool stat3 = alpha.is_ub(idx) and (dec < -1e-3);
  return (stat1 or stat2 or stat3);
}

void Tmy_G::set_kkt(Tmy_double rho, T_alpha_container alpha, T_grad_container &grad)
{
  for (int i = 0; i < _jml_data; ++i)
  {
    bool tmp = is_kkt(i, rho, alpha, grad);
    grad.set_kkt(i, tmp);

    if (!tmp)
    {
      grad.mv_idx(i, 0);
    }
    else
    {
      grad.mv_idx(i, 1);
    }
  }
}

int Tmy_G::cari_idx_a(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel)
{
  int idx_a = -1;
  auto cek_filter = [](callback_param var_b, callback_param var_a, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel) -> bool
  {
    bool is_pass = true;
    bool kondisi1 = (var_b.dec < -1e-3) and (alpha[var_b.idx] < alpha.ub());
    bool kondisi2 = (var_b.dec > 1e-3) and (alpha[var_b.idx] > alpha.lb());
    is_pass = kondisi1 or kondisi2;
    if (is_pass)
    {
      // if (alpha.is_nol(var_b.idx))
      // {
      //   is_pass = !(alpha.is_nol(var_a.idx));
      // }
      is_pass = !alpha.is_nol(var_b.idx);
    }
    return is_pass;
  };

  idx_a = max(idx_b, rho, alpha, grad, kernel, cek_filter);
  return idx_a;
}

int Tmy_G::cari_idx_lain(int idx_b, Tmy_double rho, Tmy_kernel *kernel, T_alpha_container alpha, T_grad_container grad, Tmy_alpha *my_alpha)
{

  int idx_a = -1;
  auto cek = [](callback_param var_b, callback_param var_a, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, Tmy_alpha *my_alpha) -> bool
  {
    bool is_pass = true;
    is_pass = alpha.is_sv(var_a.idx);

    if (is_pass)
    {
      Tmy_double Fb = var_b.obj;
      Tmy_double Fa = var_a.obj;

      vector<Tmy_double> hsl_eta = kernel->hit_eta(var_b.idx, var_a.idx);
      Tmy_double delta = hsl_eta[0] * (Fa - Fb);
      Treturn_is_pass hsl = my_alpha->is_pass(var_b.idx, var_a.idx, delta, alpha);

      is_pass = hsl.is_pass;
    }

    return is_pass;
  };

  auto cek1 = [](callback_param var_b, callback_param var_a, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, Tmy_alpha *my_alpha) -> bool
  {
    bool is_pass = true;

    if (alpha.is_nol(var_b.idx))
    {
      is_pass = !(alpha.is_nol(var_a.idx));
    }

    if (is_pass)
    {
      Tmy_double Fb = var_b.obj;
      Tmy_double Fa = var_a.obj;

      vector<Tmy_double> hsl_eta = kernel->hit_eta(var_b.idx, var_a.idx);
      Tmy_double delta = hsl_eta[0] * (Fa - Fb);
      Treturn_is_pass hsl = my_alpha->is_pass(var_b.idx, var_a.idx, delta, alpha);

      is_pass = hsl.is_pass;
    }

    return is_pass;
  };

  idx_a = max(idx_b, rho, alpha, grad, kernel, my_alpha, cek);
  if (idx_a != -1)
  {
    idx_a = cari(idx_b, rho, alpha, grad, kernel, my_alpha, cek);
  }
  if (idx_a != -1)
  {
    idx_a = cari(idx_b, rho, alpha, grad, kernel, my_alpha, cek1);
  }
  return idx_a;
}

bool Tmy_G::cari_idx(int &idx_b, int &idx_a, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel)
{
  auto cek_filter = [](callback_param var_b, callback_param var_a, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel) -> bool
  {
    bool is_pass = true;
    bool kondisi1 = (var_b.dec < -1e-3) and (alpha[var_b.idx] < alpha.ub());
    bool kondisi2 = (var_b.dec > 1e-3) and (alpha[var_b.idx] > alpha.lb());
    is_pass = kondisi1 or kondisi2;
    return is_pass;
  };

  auto cek_filter1 = [](callback_param var_b, callback_param var_a, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel) -> bool
  {
    bool is_pass = true;
    // is_pass = alpha.is_sv(var_a.idx);
    if (alpha.is_nol(var_b.idx))
    {
      is_pass = !(alpha.is_nol(var_a.idx));
    }
    return is_pass;
  };

  idx_b = max(rho, alpha, grad, kernel, cek_filter);
  if (idx_b != -1)
  {
    idx_a = max(idx_b, rho, alpha, grad, kernel, cek_filter1);
  }

  return ((idx_b != -1) and (idx_a != -1));
}

void Tmy_G::update_G(int idx_b, int idx_a, Treturn_is_pass tmp, Tmy_kernel *kernel, T_alpha_container &alpha, T_grad_container &grad)
{
  Tmy_double alpha_a = alpha[idx_a];
  Tmy_double alpha_b = alpha[idx_b];

  Tmy_double delta_1 = tmp.new_alpha_i - alpha_b;
  Tmy_double delta_2 = tmp.new_alpha_j - alpha_a;

  vector<Tmy_double> data_a = kernel->get_Q(idx_a);
  vector<Tmy_double> data_b = kernel->get_Q(idx_b);

  for (int i = 0; i < _jml_data; ++i)
  {
    grad[i] = grad[i] + ((data_b[i] * delta_1) + (data_a[i] * delta_2));
  }
}

int Tmy_G::max(Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, callback_type f)
{
  Tmy_double gmax = -HUGE_VAL;
  int idx_max = -1;

  vector<int> rand_idx = grad.get_rand_idx();

  for (size_t i = 0; i < _jml_data; i++)
  {
    Tmy_double obj = grad.obj(rand_idx[i], rho);
    Tmy_double dec = grad.dec(rand_idx[i], rho);

    callback_param var_b;
    var_b.idx = rand_idx[i];
    var_b.dec = dec;
    var_b.obj = obj;
    var_b.grad = grad[rand_idx[i]];

    callback_param var_a;
    bool is_pass = true;
    // is_pass = !is_kkt(i, rho, alpha, grad);
    if (is_pass)
    {
      is_pass = f(var_b, var_a, alpha, grad, kernel);
    }

    if (is_pass)
    {
      Tmy_double abs_obj = abs(obj);
      if (abs_obj >= gmax)
      {
        gmax = abs_obj;
        idx_max = rand_idx[i];
        grad.mv_idx(rand_idx[i], 0);
      }
    }
  }

  return idx_max;
}

int Tmy_G::max(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, callback_type f)
{
  Tmy_double gmax = -HUGE_VAL;
  int idx_max = -1;

  Tmy_double Gb = grad[idx_b];
  Tmy_double dec_Fb = grad.dec(idx_b, rho);
  Tmy_double obj_Fb = grad.obj(idx_b, rho);

  callback_param var_b;
  var_b.idx = idx_b;
  var_b.dec = dec_Fb;
  var_b.obj = obj_Fb;
  var_b.grad = Gb;

  Tmy_double Fa = 0.0;
  Tmy_double Fb = 0.0;

  vector<int> rand_idx = grad.get_rand_idx();

  for (int i = 0; i < _jml_data; ++i)
  {
    Tmy_double Ga = grad[i];
    Tmy_double dec_Fa = grad.dec(rand_idx[i], rho);
    Tmy_double obj_Fa = grad.obj(rand_idx[i], rho);

    callback_param var_a;
    var_a.idx = rand_idx[i];
    var_a.dec = dec_Fa;
    var_a.obj = obj_Fa;
    var_a.grad = Ga;

    Fa = obj_Fa;
    Fb = obj_Fb;

    bool is_pass = true;

    if (is_pass)
    {
      is_pass = f(var_b, var_a, alpha, grad, kernel);
    }

    if (is_pass)
    {
      Tmy_double diff_F = Fb - Fa;
      Tmy_double abs_diff_F = abs(diff_F);
      if ((abs_diff_F >= gmax))
      {
        gmax = abs_diff_F;
        idx_max = rand_idx[i];
        grad.mv_idx(rand_idx[i], 0);
      }
    }
  }
  return idx_max;
}

int Tmy_G::max(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, Tmy_alpha *my_alpha, callback_type1 f)
{
  Tmy_double gmax = -HUGE_VAL;
  int idx_max = -1;

  Tmy_double Gb = grad[idx_b];
  Tmy_double dec_Fb = grad.dec(idx_b, rho);
  Tmy_double obj_Fb = grad.obj(idx_b, rho);

  callback_param var_b;
  var_b.idx = idx_b;
  var_b.dec = dec_Fb;
  var_b.obj = obj_Fb;
  var_b.grad = Gb;

  Tmy_double Fa = 0.0;
  Tmy_double Fb = 0.0;

  vector<int> rand_idx = grad.get_rand_idx();

  for (int i = 0; i < _jml_data; ++i)
  {
    Tmy_double Ga = grad[i];
    Tmy_double dec_Fa = grad.dec(rand_idx[i], rho);
    Tmy_double obj_Fa = grad.obj(rand_idx[i], rho);

    callback_param var_a;
    var_a.idx = rand_idx[i];
    var_a.dec = dec_Fa;
    var_a.obj = obj_Fa;
    var_a.grad = Ga;

    Fa = obj_Fa;
    Fb = obj_Fb;

    bool is_pass = true;

    if (is_pass)
    {
      is_pass = f(var_b, var_a, alpha, grad, kernel, my_alpha);
    }

    if (is_pass)
    {
      Tmy_double diff_F = Fb - Fa;
      Tmy_double abs_diff_F = abs(diff_F);
      if ((abs_diff_F >= gmax))
      {
        gmax = abs_diff_F;
        idx_max = rand_idx[i];
        grad.mv_idx(rand_idx[i], 0);
      }
    }
  }
  return idx_max;
}

int Tmy_G::cari(int idx_b, Tmy_double rho, T_alpha_container alpha, T_grad_container grad, Tmy_kernel *kernel, Tmy_alpha *my_alpha, callback_type1 f)
{
  // cout << " Cari 1 " << endl;
  int idx_a = -1;

  Tmy_double Gb = grad[idx_b];
  Tmy_double abs_Gb = abs(Gb);
  Tmy_double dec_Fb = grad.dec(idx_b, rho);
  Tmy_double obj_Fb = grad.obj(idx_b, rho);

  callback_param var_b;
  var_b.idx = idx_b;
  var_b.dec = dec_Fb;
  var_b.obj = obj_Fb;
  var_b.grad = Gb;

  Tmy_double Fa = 0.0;
  Tmy_double Fb = 0.0;

  vector<int> tmp_idx = grad.get_rand_idx();
  for (int i = 0; i < tmp_idx.size(); ++i)
  {
    Tmy_double Ga = grad[tmp_idx[i]];
    Tmy_double abs_Ga = abs(Ga);
    Tmy_double dec_Fa = grad.dec(tmp_idx[i], rho);
    Tmy_double obj_Fa = grad.obj(tmp_idx[i], rho);

    callback_param var_a;
    var_a.idx = tmp_idx[i];
    var_a.dec = dec_Fa;
    var_a.obj = obj_Fa;
    var_a.grad = Ga;

    Fa = obj_Fa;
    Fb = obj_Fb;

    bool is_pass = true;

    if (is_pass)
    {
      is_pass = f(var_b, var_a, alpha, grad, kernel, my_alpha);
    }

    if (is_pass)
    {
      idx_a = tmp_idx[i];
      break;
    }
  }

  return idx_a;
}

bool Tmy_G::delta_filter(int idx_b, int idx_a, T_alpha_container alpha, Tmy_double delta)
{
  bool is_pass = true;

  if (alpha.is_nol(idx_b))
  {
    is_pass = (delta > 0.0) and (delta <= alpha[idx_a]);
  }
  else
  {
    if (alpha[idx_b] > 0.0)
    {
      if (delta > 0.0)
      {
        is_pass = (delta <= alpha[idx_a]);
      }
      else
      {
        if (delta < 0.0)
        {
          is_pass = (abs(delta) <= alpha[idx_b]);
        }
      }
    }
  }

  return is_pass;
}
