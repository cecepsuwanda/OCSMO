#include <string>
#include "Tmy_list_G.h"
#include "Tmy_kernel.h"
#include "Tmy_alpha.h"
#include "Tmy_double.h"

using namespace std;

#ifndef Included_Tmy_G_H

#define Included_Tmy_G_H

class Tmy_G
{
private:
	int _jml_data;
	Tmy_list_G *_my_list_G;
	Tmy_kernel *_kernel;
    Tmy_alpha *_alphas;
public:
	Tmy_G(int jml_data,Tmy_kernel *kernel,Tmy_alpha *alphas);
	~Tmy_G();

    void clear_container();

	void init();
	Tmy_double update_rho(int idx_a,int idx_b);
	Tmy_list_G* get_list_G();
};

#endif