#include <string>
#include <math.h>
#include "Tdataframe.h"
#include "Tmy_double.h"
#include "Tmy_cache.h"
#include "global.h"

using namespace std;

#ifndef Included_Tmy_kernel_H

#define Included_Tmy_kernel_H

struct Treturn_data
{
	int idx_map=-1;
	int idx_vec=-1;
	Tmy_double data;
};


class Tmy_kernel
{
private:
	Tdataframe *_df;
	Tmy_cache *_cache;
	Tmy_double _gamma;
	int _jml_data;
	map<int, Tmy_double> _x_square;
	map<int, int> _map_swap;

	Tmy_double dot(vector<string> x,vector<string> y);
	Tmy_double kernel_function(int i,int j);

	static Treturn_data thread_hit_data(int idx_map,int idx_vec,vector<string> x,vector<string> y,Tmy_double _x_square_x,Tmy_double _x_square_y,double gamma);
	static Treturn_data thread_hit_x_square(int idx_map,vector<string> x);
public:
	Tmy_kernel(Tdataframe &df,double gamma);
	~Tmy_kernel();

	void clear_container();

	vector<Tmy_double> get_Q(int i,int size);
	vector<Tmy_double> hit_eta(int i,int j,int size);  
	Tmy_double kernel_function_f(vector<string> x,vector<string> y);
	void swap_index(int i,int j);
};

#endif