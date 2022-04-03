#include <string>
#include <iostream>
#include <cmath>
#include "global.h"
using namespace std;


#ifndef Included_Tmy_double_H

#define Included_Tmy_double_H

class Tmy_double
{

friend ostream& operator<<(ostream& os, const Tmy_double& rhs){
    os << rhs._val;
    return os;
}

friend Tmy_double operator * (const double& lhs,const Tmy_double& rhs)
{
   return (lhs*rhs._val); 
}	

friend Tmy_double operator / (const double& lhs,const Tmy_double& rhs)
{
   return (lhs/rhs._val); 
}


friend Tmy_double operator - (const double& lhs,const Tmy_double& rhs)
{
   double tmp = lhs;
   double tmp1 = rhs._val;   
   return (tmp-tmp1); 
}

friend double operator + (const double& lhs,const Tmy_double& rhs)
{
   double tmp = lhs;
   double tmp1 = rhs._val;   
   return tmp+tmp1; 
}


private:
   double _val;

   

public:
	Tmy_double();
	~Tmy_double();

	

	Tmy_double(const Tmy_double &t)
	{
		_val = t._val;		
	}
	
	Tmy_double(const double t)
	{
		double tmp = t;
		tmp = tmp;
		_val = tmp;		
	}


	Tmy_double& operator = (const Tmy_double &t)
	{
		this->_val = t._val;		
		return *this;
	}
	
	Tmy_double& operator = (const double t)
	{
		double tmp = t;
		tmp = tmp;
		this->_val = tmp;		
		return *this;
	}

   Tmy_double operator * (const Tmy_double& rhs) 
	{  
      return (_val*rhs._val);
	}

	Tmy_double operator * (const double& rhs) 
	{  
      return (_val*rhs);
	}

	Tmy_double operator / (const Tmy_double& rhs) 
	{  
      return (_val/rhs._val);
	}

	Tmy_double operator / (const double& rhs) 
	{  
      return (_val/rhs);
	}

	Tmy_double operator + (const Tmy_double& rhs) 
	{  
      return (_val+rhs._val);
	}

	Tmy_double operator + (const double& rhs) 
	{  
      return (_val+rhs);
	}

	Tmy_double operator - (const Tmy_double& rhs) 
	{  
      return (_val-rhs._val);
	}

	Tmy_double operator - (const double& rhs) 
	{  
      return (_val-rhs);
	}


	
	bool operator !=(const Tmy_double& rhs) const
	{
      double tmp  =  rhs._val;
      double tmp1 =  _val;
      tmp=tmp1-tmp; 
		return tmp != 0;
	}
	
	bool operator !=(const double& rhs) const
	{
      double tmp  =  rhs;
      double tmp1 =  _val;
      tmp=tmp1-tmp; 
		return tmp != 0;
	}

	bool operator <(const Tmy_double& rhs) const
	{
        
		return _val < rhs._val;
	}	

	bool operator <(const double& rhs) const
	{		
		return _val < rhs;
	}

	bool operator >(const Tmy_double& rhs) const
	{        
		return _val > rhs._val;
	}
	
	bool operator >(const double& rhs) const
	{		
		return _val > rhs;
	}

	bool operator >=(const Tmy_double& rhs) const
	{
        double tmp  =  rhs._val;
        double tmp1 =  _val;
        tmp1=tmp1-tmp;		
		return (_val > rhs._val) or (tmp1==0);
	}

	bool operator >=(const double& rhs) const
	{		
		
		double tmp  =  rhs;
      double tmp1 =  _val;
      tmp=tmp1-tmp;
		return (_val > rhs) or (tmp==0);
	}

	bool operator <=(const Tmy_double& rhs) const
	{
        double tmp  =  rhs._val;
        double tmp1 =  _val;
        tmp=tmp1-tmp;		
		  return (_val < rhs._val) or (tmp==0);
	}

	bool operator <=(const double& rhs) const
	{		
		  double tmp  =  rhs;
        double tmp1 =  _val;
        tmp=tmp1-tmp;
		  return (_val < rhs) or (tmp==0);
	}

	bool operator ==(const Tmy_double& rhs) const
	{
        double tmp  =  rhs._val;
        double tmp1 =  _val;
        tmp=tmp1-tmp;		
		return (tmp==0);
	}

	
	bool operator ==(const double& rhs) const
	{		
		  double tmp  =  rhs;
        double tmp1 =  _val;
        tmp=tmp1-tmp;
		  return (tmp==0);
	}


	operator double()
	{
		return _val;
	}

	operator int()
	{
		return (int) _val;
	}
	
};




#endif