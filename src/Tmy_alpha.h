#include <iostream>
#include "T_alpha_container.h"
#include "Tmy_double.h"
#include "global.h"

using namespace std;

#ifndef Included_Tmy_alpha_H

#define Included_Tmy_alpha_H

struct Treturn_is_pass
{
public:
  bool is_pass;
  Tmy_double alpha_i;
  Tmy_double alpha_j;
  Tmy_double new_alpha_i;
  Tmy_double new_alpha_j;
  Tmy_double lb;
  Tmy_double ub;

  bool operator==(const Treturn_is_pass &rhs) const
  {
    return ((new_alpha_i == rhs.new_alpha_i) and (new_alpha_j == rhs.new_alpha_j));
  }

  bool operator!=(const Treturn_is_pass &rhs) const
  {
    return ((new_alpha_i != rhs.new_alpha_i) or (new_alpha_j != rhs.new_alpha_j));
  }

  Treturn_is_pass operator-(const Treturn_is_pass &rhs)
  {
    Treturn_is_pass tmp;
    tmp.alpha_i = alpha_i - rhs.alpha_i;
    tmp.alpha_j = alpha_j - rhs.alpha_j;
    tmp.new_alpha_i = new_alpha_i - rhs.new_alpha_i;
    tmp.new_alpha_j = new_alpha_j - rhs.new_alpha_j;
    return tmp;
  }

  void set(int flag, Tmy_double val)
  {

    if (flag == 0)
    {
      Tmy_double tmp_i = new_alpha_i - val;
      Tmy_double tmp_j = new_alpha_j + tmp_i;
      tmp_i = val;

      new_alpha_i = limit_alpha(tmp_i);
      new_alpha_j = limit_alpha(tmp_j);
    }
    else
    {
      if (flag == 1)
      {
        Tmy_double tmp_j = new_alpha_j - val;
        Tmy_double tmp_i = new_alpha_i + tmp_j;
        tmp_j = val;

        new_alpha_i = limit_alpha(tmp_i);
        new_alpha_j = limit_alpha(tmp_j);
      }
    }
  }

private:
  Tmy_double limit_alpha(Tmy_double alpha)
  {
    if (alpha > ub)
    {
      alpha = ub;
    }
    else
    {
      if (alpha < lb)
      {
        alpha = lb;
      }
    }
    return alpha;
  }
};

class Tmy_alpha
{
private:
  Tconfig *_config;
  
  vector<Tmy_double> calculateBoundaries(int i, int j, T_alpha_container alpha);
  vector<Tmy_double> limit_alpha(Tmy_double alpha_a, Tmy_double alpha_b, Tmy_double Low, Tmy_double High, int flag);
  vector<Tmy_double> calculateNewAlpha(int i, int j, Tmy_double delta, Tmy_double Low, Tmy_double High, T_alpha_container alpha);

public:
  Tmy_alpha(Tconfig *v_config);
  ~Tmy_alpha();
  void init(int jml_data, T_alpha_container &alpha);
  Treturn_is_pass is_pass(int i, int j, Tmy_double delta, T_alpha_container alpha);  
};

#endif