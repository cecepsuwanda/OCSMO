
#include "Tmy_dttype.h"
#include "Tlabel_stat.h"
#include "global.h"
#include <string>

using namespace std;


#ifndef Included_Tbelow_above_H

#define Included_Tbelow_above_H

class Tbelow_above
{
 private: 
  Tmy_dttype _value;
  Tlabel_stat _below;
  Tlabel_stat _above;
 
  Tconfig* config;

public:
	Tbelow_above();
	Tbelow_above(Tconfig* v_config);
	~Tbelow_above();
	
    //void set_config(Tconfig* v_config); 

	void set_value(Tmy_dttype value);
	Tmy_dttype get_value();

	void add_below(Tlabel_stat stat);
	Tlabel_stat get_below();
	void add_above(Tlabel_stat stat);
	Tlabel_stat get_above();

	float get_overall_metric();
	float get_split_info();

	void clear();

	bool cek_valid();
	
	
};

#endif