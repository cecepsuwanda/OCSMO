#include <string>
#include <map>
#include <iterator>
#include "Tmy_double.h"

using namespace std;

#ifndef Included_Tmy_cache_H

#define Included_Tmy_cache_H

struct Treturn_is_in_head
{
	bool is_pass;
	int awal;
};

class Tmy_cache
{
private:
   int _jml_data;
   int _size;

   map<int, vector<Tmy_double>> _head;

public:
	Tmy_cache(int jml_data,int size);
	~Tmy_cache();

	Treturn_is_in_head is_in_head(int idx,int size);
	vector<Tmy_double> get_head(int idx);
	void isi_head(int idx_map,int idx_vec,Tmy_double val);

	void swap_index(int i,int j);
	
};

#endif