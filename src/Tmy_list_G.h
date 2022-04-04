#include <string>
#include <thread>
#include "Tmy_kernel.h"
#include "Tmy_list_alpha.h"
#include "Tmy_double.h"
#include "global.h"

using namespace std;

#ifndef Included_Tmy_list_G_H

#define Included_Tmy_list_G_H

struct Treturn_cari_idx
{
   int idx_b;
   int idx_a;
   Tmy_double gmax;
   Tmy_double gmin;	
};

class Tmy_list_G
{
private:
	int _jml_data;
	int _active_size;
	Tmy_kernel *_kernel;
   Tmy_list_alpha *_alpha;

   bool _unshrink;

   vector<Tmy_double> _arr_G;
   vector<Tmy_double> _arr_G_bar;
   vector<int> _active_set;        

   bool be_shrunk(int i, Tmy_double gmax1, Tmy_double gmax2);
   void swap_index(int i,int j);
public:
	Tmy_list_G(int jml_data,Tmy_kernel *kernel,Tmy_list_alpha *alpha);
	~Tmy_list_G();
	
   

   void init();
	void update_G(int idx_b,int idx_a,Tmy_double new_alpha_b,Tmy_double new_alpha_a);
	Tmy_double get_G(int idx);
	Treturn_cari_idx cari_idx(Tmy_double rho);
	int cari_idx_lain(int idx_b,Tmy_double rho);
	bool is_kkt(int idx,Tmy_double rho);
   
   void do_shrinking();
   void reconstruct_gradient();

   int get_active_size();
   void reset_active_size();

   void reverse_swap();


	
};

#endif