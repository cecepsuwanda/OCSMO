#include "Tmy_cache.h"



Tmy_cache::Tmy_cache(int jml_data,int size)
{
   _jml_data = jml_data;
   _size = size;
}

Tmy_cache::~Tmy_cache()
{
   clear_container();
}

void Tmy_cache::clear_container()
{
  for (auto& elemen : _head)
   {
      elemen.second.clear();
   }

   _head.clear(); 
}

Treturn_is_in_head Tmy_cache::is_in_head(int idx,int size)
{
  Treturn_is_in_head hasil;
  hasil.is_pass = false;
  hasil.awal = 0;
  map<int,vector<Tmy_double>>::iterator it;  
  it = _head.find(idx);
  if (it != _head.end())
  {   
    if(_head[idx].size()<size)
    {
       hasil.awal = _head[idx].size();
       for (int i = _head[idx].size(); i < size; ++i)
       {
          _head[idx].push_back(0.0);
       }
    }else{
       hasil.is_pass = true;
    }
    
  }else{
     if(_head.size()==_size){       
       _head.erase (_head.begin()); 
     }
    _head[idx].reserve(size);
    _head[idx].assign(size,0.0);    
  }
  return hasil;
}

vector<Tmy_double> Tmy_cache::get_head(int idx)
{
   return _head[idx];
}

void Tmy_cache::isi_head(int idx_map,int idx_vec,Tmy_double val)
{   
   _head[idx_map][idx_vec]=val;   
      
}

