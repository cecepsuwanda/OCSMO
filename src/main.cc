#include <iostream>
#include <experimental/filesystem>
#include <chrono>
#include "Tconf_metrix.h"
#include "Tmy_svm.h"

using namespace std;
using std::experimental::filesystem::directory_iterator;
using std::experimental::filesystem::exists;
using std::experimental::filesystem::path;

void baca_data(Tdataframe &df, string sumber_data, string tipe_data)
{
  df.by_pass_filter_on();
  df.read_data(sumber_data);
  df.read_data_type(tipe_data);
  // df.info();
}

void isi_conf_matrix(Tconf_metrix &conf_metrix, vector<string> label, vector<string> prediksi)
{
  conf_metrix.add_konversi_asli("known", "inside");
  conf_metrix.add_konversi_asli("normal", "inside");
  conf_metrix.add_konversi_asli("unknown", "outside");

  for (int i = 0; i < prediksi.size(); ++i)
  {
    conf_metrix.add_jml(label[i], prediksi[i], 1);
  }

  conf_metrix.kalkulasi();
}

void cetak_conf_matrix(Tconf_metrix &conf_metrix)
{

  cout << conf_metrix << endl;
}

struct Twaktu_proses
{
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point finish;
  chrono::duration<double> elapsed;
  time_t tanggal_mulai;
  time_t tanggal_selesai;
  void mulai()
  {
    start = chrono::high_resolution_clock::now();
    chrono::time_point<std::chrono::system_clock> now = chrono::system_clock::now();
    tanggal_mulai = chrono::system_clock::to_time_t(now);
  }

  void selesai()
  {
    finish = chrono::high_resolution_clock::now();
    elapsed = finish - start;
    chrono::time_point<std::chrono::system_clock> now = chrono::system_clock::now();
    tanggal_selesai = chrono::system_clock::to_time_t(now);
  }

  void cetak()
  {
    cout << "Elapsed time: " << elapsed.count() << " s\n";
    cout << "Mulai : " << put_time(std::localtime(&tanggal_mulai), "%F %T.\n");
    cout << "Selesai : " << put_time(std::localtime(&tanggal_selesai), "%F %T.\n");
  }
};

int main(int argc, char *argv[])
{
  char *endptr;
  Tconfig config;
  config.f_datatype = argv[2];
  config.path_model = argv[1];
  // config.svm_path = config.path_model + "/" + argv[3];

  double bb_gamma = strtod(argv[3], &endptr);
  double ba_gamma = strtod(argv[4], &endptr);

  double bb_V = strtod(argv[5], &endptr);
  double ba_V = strtod(argv[6], &endptr);

  config.gamma = bb_gamma;
  config.V = bb_V;

  Tconf_metrix conf_metrix_train_all;
  Tconf_metrix conf_metrix_test_all;
  vector<string> label_train_all;
  vector<string> label_test_all;
  vector<string> hasil_train_max_all;
  vector<string> hasil_test_max_all;

  Twaktu_proses waktu_proses;
  waktu_proses.mulai();

  int jml = 0;

  for (const auto &file : directory_iterator(config.path_model + "/train"))
  {
    string str = file.path().filename();
    if ((str != "train_model_36.csv")) // and (str != "train_model_36.csv")
    {

      Tdataframe df_train(&config);
      baca_data(df_train, file.path(), config.f_datatype);
      vector<string> label_train = df_train.get_list_label();

      str.replace(0, 5, "test");
      string file_test = config.path_model + "/test/" + str;

      Tdataframe df_test(&config);
      baca_data(df_test, file_test, config.f_datatype);
      vector<string> label_test = df_test.get_list_label();

      // df_train.info();
      // df_test.info();

      Twaktu_proses waktu_proses;
      waktu_proses.mulai();

      double v_max = 0.0;
      double gamma_max = 0.0;
      float f1_max = -100;
      vector<string> hasil_train_max;
      vector<string> hasil_test_max;

      for (double j = bb_gamma; j <= ba_gamma; j = j + 0.0001)
      {
        for (double i = bb_V; i <= ba_V; i = i + 0.01)
        {
          config.gamma = j;
          config.V = i;

          Tmy_svm my_svm(&config);
          Treturn_train hsl_train = my_svm.train(df_train);
          vector<string> hasil_train = my_svm.test(df_train);

          cout << "file = " << str;
          cout << " iterasi = " << hsl_train.jml_iterasi;
          cout << " V = " << i;
          cout << " gamma = " << j;
          cout << " jml kkt = " << hsl_train.n_kkt;
          cout << " jml all sv = " << hsl_train.n_all_sv;
          cout << " jml alpha = " << hsl_train.jml_alpha;
          cout << " jml sv = " << hsl_train.n_sv;
          cout << " jml alpha sv = " << hsl_train.jml_alpha_n_sv;
          cout << " rho = " << hsl_train.rho;
          cout << " is optimum = " << (hsl_train.is_optimum == true ? "Yes" : "No");

          // Tconf_metrix conf_metrix_train;
          // isi_conf_matrix(conf_metrix_train, label_train, hasil_train);
          // cetak_conf_matrix(conf_metrix_train);

          vector<string> hasil_test = my_svm.test(df_test);

          Tconf_metrix conf_metrix_test;
          isi_conf_matrix(conf_metrix_test, label_test, hasil_test);
          // cetak_conf_matrix(conf_metrix_test);

          float tmp_F1 = conf_metrix_test.get_F1();
          cetak(" F1 = %f \n", tmp_F1);
          if (tmp_F1 > f1_max)
          {
            v_max = i;
            gamma_max = j;
            f1_max = tmp_F1;
            hasil_train_max = hasil_train;
            hasil_test_max = hasil_test;
          }
        }
      }
      waktu_proses.selesai();
      // waktu_proses.cetak();

      df_train.clear_memory();
      df_train.close_file();
      df_test.clear_memory();
      df_test.close_file();

      // cetak("gamma  max = %f \n", gamma_max);
      // cetak("V  max = %f \n", v_max);
      // cetak("F1 max = %f \n", f1_max);
      // Tconf_metrix conf_metrix;
      // isi_conf_matrix(conf_metrix, label_train, hasil_train_max);
      // cetak_conf_matrix(conf_metrix);

      // Tconf_metrix conf_metrix1;
      // isi_conf_matrix(conf_metrix1, label_test, hasil_test_max);
      // cetak_conf_matrix(conf_metrix1);

      label_train_all.insert(label_train_all.end(), label_train.begin(), label_train.end());
      label_test_all.insert(label_test_all.end(), label_test.begin(), label_test.end());
      hasil_train_max_all.insert(hasil_train_max_all.end(), hasil_train_max.begin(), hasil_train_max.end());
      hasil_test_max_all.insert(hasil_test_max_all.end(), hasil_test_max.begin(), hasil_test_max.end());
    }
  }
  isi_conf_matrix(conf_metrix_train_all, label_train_all, hasil_train_max_all);
  isi_conf_matrix(conf_metrix_test_all, label_test_all, hasil_test_max_all);
  cetak_conf_matrix(conf_metrix_train_all);
  cetak_conf_matrix(conf_metrix_test_all);

  waktu_proses.selesai();
  // waktu_proses.cetak();

  return 0;
}
