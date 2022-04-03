#include "Tmy_list_G.h"


Tmy_list_G::Tmy_list_G(int jml_data,Tmy_kernel *kernel,Tmy_list_alpha *alpha)
{
  _jml_data = jml_data;
  _active_size = jml_data;
  _unshrink = false;
  _kernel = kernel;
  _alpha = alpha;
  _arr_G.reserve(_jml_data);
  _arr_G.assign(_jml_data,0.0);
  _arr_G_bar.reserve(_jml_data);
  _arr_G_bar.assign(_jml_data,0.0);
  _active_set.reserve(_jml_data);
  _active_set.assign(_jml_data,0);
  init();
}

Tmy_list_G::~Tmy_list_G()
{
   _arr_G.clear();
   _arr_G_bar.clear();
   _active_set.clear();
}

void Tmy_list_G::init()
{   
   //cetak("Start init G : \n");

   for (int i = 0; i < _jml_data; ++i)
   {
       _active_set[i]=i;
   }

   vector<int> idx_alpha = _alpha->get_list_lb_ub(0);
   int i=0;
   for (auto& idx : idx_alpha)
   {   	 
   	 // if((i%100)==0){
     //    cetak(".");
     //   }
     Tmy_double alpha = _alpha->get_alpha(idx);
   	 vector<Tmy_double> data = _kernel->get_Q(idx,_jml_data);     
     for (int j = 0; j < _jml_data; ++j)
     {   	    
   	   _arr_G[j]=_arr_G[j]+(alpha*data[j]);       
     }

     if( _alpha->is_upper_bound(idx)==true)
     {
       for (int j = 0; j < _jml_data; ++j)
       {        
          _arr_G_bar[j]=_arr_G_bar[j]+(_alpha->get_ub()*data[j]);       
       }   
     } 

     i=i+1;
          
   } 
   //cetak("\nEnd init G : \n");

}

void Tmy_list_G::update_G(int idx_b,int idx_a,Tmy_double new_alpha_b,Tmy_double new_alpha_a)
{  
  
  Tmy_double alpha_a = _alpha->get_alpha(idx_a);
  Tmy_double alpha_b = _alpha->get_alpha(idx_b);

  Tmy_double delta_1 = new_alpha_b-alpha_b;
  Tmy_double delta_2 = new_alpha_a-alpha_a;

  vector<Tmy_double> data_a = _kernel->get_Q(idx_a,_active_size);
  vector<Tmy_double> data_b = _kernel->get_Q(idx_b,_active_size);   
  
  for (int i = 0; i < _active_size; ++i)
  {    
   _arr_G[i]=_arr_G[i]+((data_b[i]*delta_1)+(data_a[i]*delta_2));            
  }

  bool is_alpha_a_ub = _alpha->is_upper_bound(idx_a);
  bool is_alpha_b_ub = _alpha->is_upper_bound(idx_b);
  _alpha->update_alpha(idx_a,new_alpha_a);
  _alpha->update_alpha(idx_b,new_alpha_b);

  if(is_alpha_a_ub!=_alpha->is_upper_bound(idx_a))
  {
     vector<Tmy_double> data_a = _kernel->get_Q(idx_a,_jml_data);
     if(is_alpha_a_ub==true)
     {
        for (int i = 0; i < _jml_data; ++i)
        {    
           _arr_G_bar[i]=_arr_G_bar[i]-(data_a[i]*_alpha->get_ub());            
        }
     }else{
        for (int i = 0; i < _jml_data; ++i)
        {    
           _arr_G_bar[i]=_arr_G_bar[i]+(data_a[i]*_alpha->get_ub());            
        }
     }
  }

  if(is_alpha_b_ub!=_alpha->is_upper_bound(idx_b))
  {
    vector<Tmy_double> data_b = _kernel->get_Q(idx_b,_jml_data);
    if(is_alpha_b_ub==true)
     {
        for (int i = 0; i < _jml_data; ++i)
        {    
           _arr_G_bar[i]=_arr_G_bar[i]-(data_b[i]*_alpha->get_ub());            
        } 
     }else{
        for (int i = 0; i < _jml_data; ++i)
        {    
           _arr_G_bar[i]=_arr_G_bar[i]+(data_b[i]*_alpha->get_ub());            
        }
     } 
  }



}

Tmy_double Tmy_list_G::get_G(int idx)
{
  return _arr_G[idx];
}

bool Tmy_list_G::is_kkt(int idx,Tmy_double rho)
{
  Tmy_double F = _arr_G[idx]-rho;
  // if(F<1e-3)
  // {
  //   F=0.0;
  // }
  vector<bool> hsl = _alpha->is_alpha_sv(idx);
  bool stat = false;
  

  if((hsl[3]==true)and(F>=1e-3))
  {
     stat = true; 
  }else{
    if((hsl[1]==true)and((F>-1e-3) and (F<1e-3)))
    {
       stat = true; 
    }else{
       if((hsl[2]==true)and(F<=1e-3))
       {
         stat = true; 
       }
    } 
  }

  return stat;  
}

Treturn_cari_idx Tmy_list_G::cari_idx(Tmy_double rho)
{
   vector<int> idx_alpha_not_lb = _alpha->get_list_lb_ub(0);
   vector<int> idx_alpha_not_ub = _alpha->get_list_lb_ub(1);
   
   int idx_b = -1,idx_a=-1;
   Tmy_double gmax = -HUGE_VAL,kkt_max=-HUGE_VAL,gmin=-HUGE_VAL,diff_max=-HUGE_VAL,obj_diff_min=HUGE_VAL;

   for (auto& idx : idx_alpha_not_ub)
    {      
       if(idx<_active_size){  
         Tmy_double Fb = _arr_G[idx];       
         Tmy_double diff = Fb;
         Tmy_double tmp = -1.0*diff;         
         if (tmp>=gmax)
         {
           gmax = tmp;           
           idx_b = idx;
           _alpha->mv_lb_ub(idx,1);                      
         }
       }            
    } 

   if(idx_b!=-1)
   {
      Tmy_double Fb = _arr_G[idx_b];
      for (auto& idx : idx_alpha_not_lb)
      {       
        if(idx<_active_size){  
          Tmy_double Fa = _arr_G[idx];
          Tmy_double diff = Fa;                   
          if (diff>=gmin)
          {
            gmin = diff;
            _alpha->mv_lb_ub(idx,0);
          }

          Tmy_double grad_diff = gmax+Fa;           
          if (grad_diff>1e-3)
          {
             _alpha->mv_lb_ub(idx,0);
             Tmy_double obj_diff;
             vector<Tmy_double> tmp_hsl = _kernel->hit_eta(idx,idx_b,_active_size);
             Tmy_double quad_coef = tmp_hsl[0];
             obj_diff = -1.0*(grad_diff*grad_diff)*quad_coef;
             if(obj_diff<=obj_diff_min)
             {
                obj_diff_min=obj_diff;
                idx_a = idx;
                _alpha->mv_lb_ub(idx,0);
             }            
          }
        }              
     } 
   }   
   return {idx_b,idx_a,gmax,gmin};
}

int Tmy_list_G::cari_idx_lain(int idx_b,Tmy_double rho)
{
   vector<int> idx_alpha_not_lb = _alpha->get_list_lb_ub(0);
   vector<int> idx_alpha_not_ub = _alpha->get_list_lb_ub(1);
      
   int idx_a = -1;

   for (auto& idx : idx_alpha_not_lb)
   {
     if(idx<_active_size){
       Tmy_double Fb = _arr_G[idx_b];
       Tmy_double Fa = _arr_G[idx];
       // Tmy_double gmax = -1.0*Fb;
       // Tmy_double gmin = Fa;
       // Tmy_double diff = gmax+gmin;
       // if(diff>0.0){
           vector<Tmy_double> hsl_eta = _kernel->hit_eta(idx_b,idx,_active_size);
         
           Tmy_double delta = (Fa-Fb)*hsl_eta[0];
         
           vector<Tmy_double> alpha;
           Treturn_is_pass tmp = _alpha->is_pass(idx_b,idx,delta);
           if(tmp.is_pass==true)
           {
              idx_a = idx;
              break;
           }
       //}    
    }
   }

   if(idx_a==-1)
   {
     for (auto& idx : idx_alpha_not_ub)
     {
        if(idx<_active_size){
          Tmy_double Fb = _arr_G[idx_b];
          Tmy_double Fa = _arr_G[idx];
          // Tmy_double gmax = -1.0*Fb;
          // Tmy_double gmin = Fa;
          // Tmy_double diff = gmax+gmin;
          // if(diff>0.0){
              vector<Tmy_double> hsl_eta = _kernel->hit_eta(idx_b,idx,_active_size);
         
              Tmy_double delta = (Fa-Fb)*hsl_eta[0];
         
              vector<Tmy_double> alpha;
              Treturn_is_pass tmp = _alpha->is_pass(idx_b,idx,delta);
              if(tmp.is_pass==true)
              {
                 idx_a = idx;
                 break;
              }
          //}    
        }
     }
   }


   return idx_a;

}

bool Tmy_list_G::be_shrunk(int i, Tmy_double gmax1, Tmy_double gmax2)
{
  if(_alpha->is_upper_bound(i))
    {
       return((-1.0*_arr_G[i]) > gmax1);        
    }
    else if(_alpha->is_lower_bound(i))
    {
       return(_arr_G[i] > gmax2);        
    }
    else
        return(false);  
}

void Tmy_list_G::swap_index(int i,int j)
{
    
    _kernel->swap_index(i,j);    
    _alpha->swap_index(i,j);    
    swap(_arr_G[i],_arr_G[j]);
    swap(_active_set[i],_active_set[j]);
    swap(_arr_G_bar[i],_arr_G_bar[j]); 
}

void Tmy_list_G::reverse_swap()
{
    for (int i = 0; i < _active_set.size(); ++i)
    {
        if(_active_set[i]!=i)
        {
          _alpha->swap_index(i,_active_set[i]);
          swap(_arr_G[i],_arr_G[_active_set[i]]);          
          _active_set[_active_set[i]]=_active_set[i];
          _active_set[i]=i;  
        }
    }
}


void Tmy_list_G::do_shrinking()
{
  cout<<"do_shrinking"<<endl;
  Tmy_double gmax1=-HUGE_VAL,gmax2=-HUGE_VAL;
  
   for (int i = 0; i < _active_size; ++i)
   {
      if(_alpha->is_upper_bound(i)==false)
      {
        if((-1.0*_arr_G[i])>=gmax1)
        {
           gmax1 = (-1.0*_arr_G[i]);
        } 
      } 

      if(_alpha->is_lower_bound(i)==false)
      {
        if(_arr_G[i]>=gmax2)
        {
           gmax2 = _arr_G[i];
        } 
      }
   }

   if(_unshrink == false && ((gmax1 + gmax2) <= (1e-3*10)))
    {
        cout<<"un shrink"<<endl;
        _unshrink = true;
        reconstruct_gradient();
        _active_size = _jml_data;        
    }

    for(int i=0;i<_active_size;i++)
    {
        if (be_shrunk(i, gmax1, gmax2))
        {
            _active_size--;
            while (_active_size > i)
            {
                if (!be_shrunk(_active_size, gmax1, gmax2))
                {
                    swap_index(i,_active_size);
                    break;
                }
                _active_size--;
            }
        }
    }
    cout<< "_active_size " <<_active_size << " _jml_data "<<_jml_data<<endl;
}

void Tmy_list_G::reconstruct_gradient()
{
  if(_active_size!=_jml_data){
    cout<<"start reconstruct gradient"<<endl;

    int i,j;
    int nr_free = 0;

    for(j=_active_size;j<_jml_data;j++)
        _arr_G[j] = _arr_G_bar[j];

    for(j=0;j<_active_size;j++)
        if(_alpha->is_free(j))
            nr_free++;
    

    if (nr_free*_jml_data > 2*_active_size*(_jml_data-_active_size))
    {
        for(i=_active_size;i<_jml_data;i++)
        {
            vector<Tmy_double> Q_i = _kernel->get_Q(i,_active_size);
            for(j=0;j<_active_size;j++)
                if(_alpha->is_free(j)){
                    Tmy_double alpha_j = _alpha->get_alpha(j);
                    _arr_G[i] = _arr_G[i] + (alpha_j * Q_i[j]);
                }
        }
    }
    else
    {
        for(i=0;i<_active_size;i++)
            if(_alpha->is_free(i))
            {                
                vector<Tmy_double> Q_i = _kernel->get_Q(i,_jml_data);                
                Tmy_double alpha_i = _alpha->get_alpha(i);
                for(j=_active_size;j<_jml_data;j++)
                    _arr_G[j] = _arr_G[j] + (alpha_i * Q_i[j]);
            }
    }

    cout<<"end reconstruct gradient"<<endl;
  } 
}

int Tmy_list_G::get_active_size()
{

  return _active_size;  
}

void Tmy_list_G::reset_active_size()
{
  _active_size = _jml_data;
}   
