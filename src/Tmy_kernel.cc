#include "Tmy_kernel.h"

Tmy_kernel::Tmy_kernel(Tdataframe &df, double gamma)
{
    _df = &df;
    _gamma = gamma;
    _jml_data = _df->getjmlrow_svm();

    _cache = new Tmy_cache(_jml_data, 1000);

    vector<future<Treturn_data>> async_worker;
    for (int i = 0; i < _jml_data; ++i)
    {
        vector<string> x_i = _df->goto_rec(i);
        async_worker.push_back(async(std::launch::async, &Tmy_kernel::thread_hit_x_square, i, x_i));

        if ((async_worker.size() % 10) == 0)
        {
            Treturn_data hsl;
            for (future<Treturn_data> &th : async_worker)
            {
                hsl = th.get();
                _x_square[hsl.idx_map] = hsl.data;
            }

            async_worker.clear();
            async_worker.shrink_to_fit();
        }
    }

    if (async_worker.size() > 0)
    {
        Treturn_data hsl;
        for (future<Treturn_data> &th : async_worker)
        {
            hsl = th.get();
            _x_square[hsl.idx_map] = hsl.data;
        }

        async_worker.clear();
        async_worker.shrink_to_fit();
    }
}

Tmy_kernel::~Tmy_kernel()
{
    clear_container();
}

void Tmy_kernel::clear_container()
{
    _x_square.clear();
    _map_swap.clear();

    _cache->clear_container();
}

Tmy_double Tmy_kernel::dot(vector<string> x, vector<string> y)
{
    int jml = x.size();
    Tmy_double sum = 0;
    for (int i = 0; i < jml; ++i)
    {
        sum = sum + (stod(x[i]) * stod(y[i]));
    }
    return sum;
}

Tmy_double Tmy_kernel::kernel_function(int i, int j)
{

    int idx_df = i;
    // map<int, int>::iterator it;
    // it = _map_swap.find(idx_df);
    // if (it != _map_swap.end())
    // {
    //     idx_df = _map_swap[idx_df];
    // }
    vector<string> x_i = _df->goto_rec(idx_df);

    idx_df = j;
    // it = _map_swap.find(idx_df);
    // if (it != _map_swap.end())
    // {
    //     idx_df = _map_swap[idx_df];
    // }

    vector<string> x_j = _df->goto_rec(idx_df);

    double tmp1 = (_x_square[i] + _x_square[j] - 2.0 * dot(x_i, x_j));
    double tmp2 = (-1.0 * _gamma);
    Tmy_double tmp = exp(tmp2 * tmp1);
    return tmp;
}

Treturn_data Tmy_kernel::thread_hit_x_square(int idx_map, vector<string> x)
{
    int jml = x.size();

    double sum = 0.0;
    for (int i = 0; i < jml; ++i)
    {
        sum = sum + (stod(x[i]) * stod(x[i]));
    }

    Tmy_double hasil = sum;
    return {idx_map, -1, hasil};
}

Treturn_data Tmy_kernel::thread_hit_data(int idx_map, int idx_vec, vector<string> x, vector<string> y, Tmy_double _x_square_x, Tmy_double _x_square_y, double gamma)
{
    int jml = x.size();

    Tmy_double sum1 = _x_square_y;

    // double sum1 = 0.0;
    // for (int i = 0; i < jml; ++i)
    // {
    //   sum1=sum1+(stod(y[i])*stod(y[i]));
    // }

    Tmy_double sum2 = _x_square_x;

    // double sum2 = 0.0;
    // for (int i = 0; i < jml; ++i)
    // {
    //   sum2=sum2+(stod(x[i])*stod(x[i]));
    // }

    double sum3 = 0.0;
    for (int i = 0; i < jml; ++i)
    {
        sum3 = sum3 + (stod(x[i]) * stod(y[i]));
    }

    double tmp = -1.0 * gamma * (sum1 + sum2 - 2.0 * sum3);
    Tmy_double hasil = exp(tmp);
    return {idx_map, idx_vec, hasil};
}

vector<Tmy_double> Tmy_kernel::get_Q(int i)
{
    Treturn_is_in_head hasil = _cache->is_in_head(i, _jml_data);

    if (hasil.is_pass == false)
    {
        int idx_df = i;
        // map<int, int>::iterator it;
        // it = _map_swap.find(idx_df);
        // if (it != _map_swap.end())
        // {
        //     idx_df = _map_swap[idx_df];
        // }

        vector<string> x_j = _df->goto_rec(idx_df);

        vector<future<Treturn_data>> async_worker;
        for (int j = hasil.awal; j < _jml_data; ++j)
        {
            // cetak("*");

            int idx_df = j;
            // map<int, int>::iterator it;
            // it = _map_swap.find(idx_df);
            // if (it != _map_swap.end())
            // {
            //     idx_df = _map_swap[idx_df];
            // }

            vector<string> x_i = _df->goto_rec(idx_df);

            async_worker.push_back(async(std::launch::async, &Tmy_kernel::thread_hit_data, i, j, x_i, x_j, _x_square[j], _x_square[i], _gamma));

            if ((async_worker.size() % 10) == 0)
            {
                Treturn_data hsl;
                for (future<Treturn_data> &th : async_worker)
                {
                    hsl = th.get();
                    _cache->isi_head(hsl.idx_map, hsl.idx_vec, hsl.data);
                }

                async_worker.clear();
                async_worker.shrink_to_fit();
            }

            //_cache->isi_head(i,j,kernel_function(j,i));
        }

        if (async_worker.size() > 0)
        {
            Treturn_data hsl;
            for (future<Treturn_data> &th : async_worker)
            {
                hsl = th.get();
                _cache->isi_head(hsl.idx_map, hsl.idx_vec, hsl.data);
            }

            async_worker.clear();
            async_worker.shrink_to_fit();
        }
    }
    vector<Tmy_double> data = _cache->get_head(i);
    return data;
}

vector<Tmy_double> Tmy_kernel::get_diff_Q(int i, int j)
{
    vector<Tmy_double> hasil;
    hasil.reserve(_jml_data);
    hasil.assign(_jml_data, 0.0);
    for (int k = 0; k < _jml_data; ++k)
    {
        vector<Tmy_double> tmp = get_Q(k);
        hasil[k] = (tmp[j] - tmp[i]);
    }

    return hasil;
}

vector<Tmy_double> Tmy_kernel::get_rho_Q(int i)
{
    vector<Tmy_double> hasil;
    hasil.reserve(_jml_data);
    hasil.assign(_jml_data, 0.0);
    for (int k = 0; k < _jml_data; ++k)
    {
        vector<Tmy_double> tmp = get_Q(k);
        hasil[k] = tmp[i];
    }

    return hasil;
}

vector<Tmy_double> Tmy_kernel::hit_eta(int i, int j)
{
    vector<Tmy_double> Q_i = get_Q(i);
    vector<Tmy_double> Q_j = get_Q(j);

    Tmy_double k11 = Q_i[i]; // kernel_function(i,i);
    Tmy_double k12 = Q_i[j]; // kernel_function(j,i);
    Tmy_double k22 = Q_j[j]; // kernel_function(j,j);
    Tmy_double p_eta = k11 + k22 - (2.0 * k12);

    Tmy_double eta = 0.0;
    if (p_eta != 0.0)
    {
        eta = 1.0 / p_eta;
    }
    else
    {
        eta = 1.0 / 1e-12;
    }
    // cout<<eta<<","<<k11<<","<<k12<<","<<k22<<endl;
    return {eta, k11, k12, k22};
}

Tmy_double Tmy_kernel::kernel_function_f(vector<string> x, vector<string> y)
{

    int jml = y.size();

    double sum1 = 0.0;
    for (int i = 0; i < jml; ++i)
    {
        sum1 = sum1 + (stod(y[i]) * stod(y[i]));
    }

    double sum2 = 0.0;
    for (int i = 0; i < jml; ++i)
    {
        sum2 = sum2 + (stod(x[i]) * stod(x[i]));
    }

    double sum3 = 0.0;
    for (int i = 0; i < jml; ++i)
    {
        sum3 = sum3 + (stod(x[i]) * stod(y[i]));
    }

    double tmp = -1.0 * _gamma * (sum1 + sum2 - 2.0 * sum3);
    Tmy_double hasil = exp(tmp);
    return hasil;
}
