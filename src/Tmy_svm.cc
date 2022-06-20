#include "Tmy_svm.h"

Tmy_svm::Tmy_svm(Tconfig *v_config)
{
   _config = v_config;
   _my_alpha = new Tmy_alpha(_config);
}

Tmy_svm::~Tmy_svm()
{
   delete _my_alpha;
   delete _my_kernel;
   _model.clear();
   _alpha_sv.clear();
}

bool Tmy_svm::take_step(int idx_b, int idx_a)
{
   bool stat = false;
   if (idx_b == idx_a)
   {
   }
   else
   {
      // cout << " take step idx_b " << idx_b << " idx_a " << idx_a << endl;
      Tmy_double Fa = _grad.dec(idx_a, _rho);
      Tmy_double Fb = _grad.dec(idx_b, _rho);
      vector<Tmy_double> hsl_eta = _my_kernel->hit_eta(idx_b, idx_a);
      Tmy_double delta = hsl_eta[0] * (Fa - Fb);

      // cout << " delta " << delta << endl;

      Treturn_is_pass hsl = _my_alpha->is_pass(idx_b, idx_a, delta, _alpha);

      // cout << hsl.is_pass << " old [" << hsl.alpha_i << "," << hsl.alpha_j << "] new [" << hsl.new_alpha_i << "," << hsl.new_alpha_j << "] " << endl;

      if (!hsl.is_pass)
      {
         // cout << " not pass !!!" << endl;
      }
      else
      {
         _my_G.update_G(idx_b, idx_a, hsl, _my_kernel, _alpha, _grad);

         _alpha[idx_a] = hsl.new_alpha_j;
         _alpha[idx_b] = hsl.new_alpha_i;
         _grad.mv_idx(idx_a, 1);
         _grad.mv_idx(idx_b, 1);

         _rho = _my_G.update_rho(_my_kernel, _alpha, _grad);
         _my_G.set_kkt(_rho, _alpha, _grad);
         stat = true;
         // cout << " pass !!!" << endl;
      }
   }
   return stat;
}

bool Tmy_svm::examineExample()
{
   int idx_b = -1;
   int idx_a = -1;
   bool examineAll = false;
   bool is_pass = _my_G.cari_idx(idx_b, idx_a, _rho, _alpha, _grad, _my_kernel);
   if (idx_a != -1)
   {
      // cout << " idx_b " << idx_b << " idx_a " << idx_a << endl;
      is_pass = take_step(idx_b, idx_a);
      if (!is_pass)
      {
         idx_a = _my_G.cari_idx_lain(idx_b, _rho, _my_kernel, _alpha, _grad, _my_alpha);
         if (idx_a != -1)
         {
            is_pass = take_step(idx_b, idx_a);
            if (is_pass)
            {
               examineAll = true;
            }
            else
            {
               cout << "Out 1" << endl;
            }
         }
         else
         {
            cout << "Out 2" << endl;
         }
      }
      else
      {
         examineAll = true;
      }
   }
   else
   {
      if (idx_b != -1)
      {
         idx_a = _my_G.cari_idx_lain(idx_b, _rho, _my_kernel, _alpha, _grad, _my_alpha);
         if (idx_a != -1)
         {
            is_pass = take_step(idx_b, idx_a);
            if (is_pass)
            {
               examineAll = true;
            }
            else
            {
               cout << "Out 3" << endl;
            }
         }
         else
         {
            cout << "Out 4" << endl;
         }
      }
      else
      {
         cout << "Out 5" << endl;
      }
   }
   return examineAll;
}

int Tmy_svm::examineExample(int idx_b)
{
   int tmp = 0;
   int idx_a = _my_G.cari_idx_a(idx_b, _rho, _alpha, _grad, _my_kernel);
   if (idx_a != -1)
   {
      bool is_pass = take_step(idx_b, idx_a);
      if (!is_pass)
      {
         idx_a = _my_G.cari_idx_lain(idx_b, _rho, _my_kernel, _alpha, _grad, _my_alpha);
         if (idx_a != -1)
         {
            is_pass = take_step(idx_b, idx_a);
            if (is_pass)
            {
               tmp = 1;
            }
         }
      }
      else
      {
         tmp = 1;
      }
   }
   return tmp;
}

Treturn_train Tmy_svm::train(Tdataframe &df)
{
   // cout << " Train : " << endl;
   int jml_data = df.getjmlrow_svm();
   _my_kernel = new Tmy_kernel(df, _config->gamma);
   // cout << " Init Alpha : " << endl;
   _my_alpha->init(jml_data, _alpha);
   // cout << " Init G : " << endl;
   _my_G.init(jml_data, _my_kernel, _alpha, _grad);
   // cout << " Init rho : " << endl;
   _rho = _my_G.update_rho(_my_kernel, _alpha, _grad);
   // cout << " Init kkt : " << endl;
   _my_G.set_kkt(_rho, _alpha, _grad);

   int max_iter = jml_data * 100;
   int counter = min(jml_data, 1000) + 1;
   bool stop_iter = false;
   int jml_out = 0;

   // cout << " Cetak isi alpha : " << endl;
   // for (int i = 0; i < jml_data; ++i)
   // {
   //    cout << i << setw(15) << _alpha[i] << setw(15) << _grad[i] << setw(15) << _rho << setw(15) << _grad.get_kkt(i) << endl;
   // }

   // cout << " Mulai Optimasi : " << endl;

   int iter = 0;
   bool examineAll = true;
   while ((iter < max_iter) and !stop_iter)
   {

      if (examineAll)
      {
         int i = 0;
         bool ulangi = true;
         while ((i < counter) and ulangi)
         {
            ulangi = examineExample();
            i = i + 1;
            iter = iter + 1;
         }
         if (!ulangi)
         {
            jml_out = jml_out + 1;
            if (jml_out == 10)
            {
               cout << " jml out " << jml_out << endl;
               stop_iter = true;
            }
            else
            {
               examineAll = false;
            }
         }
      }
      else
      {
         vector<int> rand_idx = _grad.get_rand_idx();
         int jml_pass = 0;
         for (size_t i = 0; i < jml_data; i++)
         {
            jml_pass = jml_pass + examineExample(rand_idx[i]);
            iter = iter + 1;
         }

         if (jml_pass > 0)
         {
            examineAll = true;
         }
         else
         {
            stop_iter = true;
         }
      }
   }

   //_alpha.nol_kan();
   _rho = _my_G.update_rho(_my_kernel, _alpha, _grad);

   // for (int i = 0; i < jml_data; ++i)
   // {
   //    cout << i << setw(15) << _alpha[i] << setw(15) << _grad[i] << setw(15) << _rho << setw(15) << _grad.get_kkt(i) << endl;
   // }

   Treturn_train tmp_train;

   tmp_train.jml_iterasi = iter;
   tmp_train.jml_alpha = _alpha.sum();
   tmp_train.n_all_sv = _alpha.n_all_sv();
   tmp_train.n_sv = _alpha.n_sv();
   tmp_train.rho = _rho;

   int n_kkt = 0;
   int i = 0;
   _alpha_sv.reserve(tmp_train.n_all_sv);
   for (int idx = 0; idx < jml_data; ++idx)
   {
      if (_alpha[idx] != 0.0)
      {
         if (_my_G.is_kkt(idx, _rho, _alpha, _grad) == true)
         {
            n_kkt = n_kkt + 1;
         }
         vector<string> data = df.goto_rec(idx);
         _alpha_sv.push_back(_alpha[idx]);
         _model.insert(pair<int, vector<string>>(i, data));
         i = i + 1;
      }
   }
   tmp_train.n_kkt = n_kkt;

   _my_kernel->clear_container();

   return tmp_train;
}

vector<string> Tmy_svm::test(Tdataframe &df)
{
   // Tmy_list_alpha *list_alpha = _my_alpha->get_alpha();
   // map<int,Tmy_double> alpha_sv = list_alpha->get_list_alpha_sv();
   int jml_data = df.getjmlrow_svm();
   std::vector<string> hasil;
   hasil.reserve(jml_data);
   hasil.assign(jml_data, "inside");

   vector<string> tmp_data;

   int j = 0;
   df.reset_file();
   while (!df.is_eof())
   {

      tmp_data = df.get_record_svm();

      Tmy_double sum = 0.0;
      for (map<int, vector<string>>::iterator it = _model.begin(); it != _model.end(); ++it)
      {
         Tmy_double tmp = _my_kernel->kernel_function_f(tmp_data, it->second);
         Tmy_double alpha = _alpha_sv[it->first];
         tmp = alpha * tmp;
         sum = sum + tmp;
      }

      sum = sum - _rho;
      if (sum >= 0.0)
      {
      }
      else
      {
         if (sum < 0.0)
         {
            hasil[j] = "outside";
         }
      }

      df.next_record();
      j = j + 1;
   }

   return hasil;
}