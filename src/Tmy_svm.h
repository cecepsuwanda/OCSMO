#include <string>
#include <iterator>
#include <map>
#include "Tdataframe.h"
#include "Tmy_alpha.h"
#include "Tmy_kernel.h"
#include "Tmy_G.h"
#include "Tmy_list_G.h"
#include "Tmy_list_alpha.h"
#include "Tmy_double.h"

using namespace std;

#ifndef Included_Tmy_svm_H

#define Included_Tmy_svm_H

struct Treturn_train
{
   int jml_iterasi;
   int n_kkt;
   Tmy_double jml_alpha;
   int n_all_sv;
   int n_sv;
   Tmy_double jml_alpha_n_sv;
   Tmy_double rho;
   bool is_optimum;
};

struct Treturn_cari_alpha
{
   bool is_pass;
   Tmy_double idx_b;
   Tmy_double idx_a;
};

class Tmy_svm
{

private:
    Tconfig *_config;
    Tmy_alpha *_my_alpha;
    Tmy_kernel *_my_kernel;
    Tmy_G *_my_G;
    Tmy_double _rho;
    vector<vector<string>> _model;
public:
	Tmy_svm(Tconfig *v_config);
	~Tmy_svm();
	Treturn_train train(Tdataframe &df);
	vector<string> test(Tdataframe &df);
	Treturn_cari_alpha cari_idx_alpha();
	bool cari_idx_a_lain(int idx_b,int *idx_alpha);
	bool take_step(int idx_b,int idx_a);
	
};

#endif
