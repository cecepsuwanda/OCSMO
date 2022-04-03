#include "Tmy_cache.h"



Tmy_cache::Tmy_cache(int jml_data,int size)
{
   _jml_data = jml_data;
   _size = size;
}

Tmy_cache::~Tmy_cache()
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

void Tmy_cache::swap_index(int i,int j)
{
   
   if(i!=j)
   {      
      map<int,vector<Tmy_double>>::iterator it_i;  
      it_i = _head.find(i);
      bool idx_i_ada = (it_i != _head.end());
      map<int,vector<Tmy_double>>::iterator it_j;
      it_j = _head.find(j);
      bool idx_j_ada = (it_j != _head.end());

      if(idx_i_ada && idx_j_ada)
      {
         vector<Tmy_double> tmp = _head[i];
         _head[i] = _head[j];
         _head[j] = tmp; 
      }else{
        if(idx_i_ada==true && idx_j_ada==false)
        {
          vector<Tmy_double> tmp = _head[i];
          _head.erase (it_i);
          _head[j]=tmp;
        }else{
           if(idx_i_ada==false && idx_j_ada==true)
           {
             vector<Tmy_double> tmp = _head[j];
             _head.erase (it_j);
             _head[i]=tmp;
           }
        }  
      }

      if(i>j) swap(i,j);

      for (auto it = _head.begin(); it != _head.end(); ++it)
      {
         int idx = it->first;
         vector<Tmy_double> tmp = it->second;
         if(tmp.size()>i)
         {
            if(tmp.size()>j)
            {
               swap(_head[idx][i],_head[idx][j]);
            }else{
               _head.erase(it);
            }
         }
      }      
   }
}