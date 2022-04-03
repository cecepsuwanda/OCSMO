#include "Tmy_list_alpha.h"



Tmy_list_alpha::Tmy_list_alpha(int v_jml_data,Tmy_double v_lb,Tmy_double v_ub){
   _jml_data = v_jml_data;
   _lb = v_lb;
   _ub = v_ub;
   _jml_alpha=0;
   _jml_alpha_n_sv=0;
   _n_all_sv=0;
   _n_sv=0;
   
   _alpha.reserve(_jml_data);
   _alpha_status.reserve(_jml_data);
   _alpha.assign(_jml_data,0.0);   
   _alpha_status.assign(_jml_data,0);
   
}

Tmy_list_alpha::~Tmy_list_alpha(){
  _alpha.clear();
  _alpha_status.clear();
  _alpha_not_ub.clear();
  _alpha_not_lb.clear();
  _alpha_free.clear();
  _alpha_sv.clear();  
}


void Tmy_list_alpha::init(Tmy_double V,Tmy_double eps){
  //cetak("init alpha : \n");
  Tmy_double tmp = V*((double)_jml_data);  
  int jml = (int) tmp;
  
  _alpha_not_lb.reserve(jml);

  for(int idx=0;idx<jml;idx++){
  	 update_alpha(idx,_ub);
  }

  // if(jml<_jml_data)
  // {
  //   update_alpha(jml,(_ub* (double) _jml_data)*(1.0/(_jml_data-jml)));
  // } 

  if(_jml_alpha<1.0)
  {
    tmp = 1.0-_jml_alpha;       
    update_alpha(jml,tmp);
    jml=jml+1;
  }
  //else{
    // Tmy_double tmp = _ub/2.0;
    // update_alpha(jml-1,tmp);
    // update_alpha(jml,tmp);
    // jml=jml+1;
  // }

  _alpha_not_lb.reserve(_jml_data-jml);

  for(int idx=jml;idx<_jml_data;idx++){
     update_alpha(idx,0.0);
  }  
}

void Tmy_list_alpha::update_alpha(int idx,Tmy_double value)
{
	if(_alpha[idx]!=0.0){
      _jml_alpha = _jml_alpha - _alpha[idx];
    }

  vector<bool> hasil = is_alpha_sv(idx);
  
  if(hasil[0]==true){
    if(_n_all_sv!=0){
      _n_all_sv = _n_all_sv-1;
    } 
  }

  if(hasil[1]==true){
    if(_n_sv!=0){
      _jml_alpha_n_sv = _jml_alpha_n_sv - _alpha[idx];    
      _n_sv = _n_sv-1;
    }
  }   

	_alpha[idx]=value;
  _jml_alpha = _jml_alpha + _alpha[idx];

  hasil = is_alpha_sv(idx);
  
  if(hasil[0]==true){    
      _n_all_sv = _n_all_sv+1;    
  }

  if(hasil[1]==true){
      _jml_alpha_n_sv = _jml_alpha_n_sv + _alpha[idx];    
      _n_sv = _n_sv+1;    
  }   

  update_alpha_sv(idx);
  update_alpha_status(idx);
  update_lb_ub(idx);  
	

}

void Tmy_list_alpha::update_alpha_status(int idx)
{
   
   if(_alpha[idx]>=_ub)
   {
   	 //cout<<"Stat 1 "<<idx<<" "<<_alpha[idx]<<endl;
     _alpha_status[idx]=1;
   }else{      
   	  if(_alpha[idx]<=_lb)
      {
   	     //cout<<"Stat 0 "<<idx<<" "<<_alpha[idx]<<endl;
         _alpha_status[idx]=0;
      }else{
   	     //cout<<"Free "<<idx<<endl;
         _alpha_status[idx]=2;
      }
   } 	
}

void Tmy_list_alpha::update_alpha_sv(int idx)
{
  
  map<int,Tmy_double>::iterator it;  
  it = _alpha_sv.find(idx);
  if (it != _alpha_sv.end())
     _alpha_sv.erase (it);
  
  vector<bool> hasil = is_alpha_sv(idx);
  if(hasil[0]==true){
     _alpha_sv[idx]=_alpha[idx];  
  }
}

void Tmy_list_alpha::update_lb_ub(int idx_cari)
{
  for(int i=0;i<_alpha_not_ub.size();i++){
    if(_alpha_not_ub[i]==idx_cari){
      _alpha_not_ub.erase(_alpha_not_ub.begin()+i);
      break;
    } 
  }

  for(int i=0;i<_alpha_not_lb.size();i++){
    if(_alpha_not_lb[i]==idx_cari){
      _alpha_not_lb.erase(_alpha_not_lb.begin()+i);
      break;
    } 
  }

  for(int i=0;i<_alpha_free.size();i++){
    if(_alpha_free[i]==idx_cari){
      _alpha_free.erase(_alpha_free.begin()+i);
      break;
    } 
  }
  
  if (is_upper_bound(idx_cari)==false)
  {
    _alpha_not_ub.push_back(idx_cari);
  }

  if (is_lower_bound(idx_cari)==false)
  {
    _alpha_not_lb.push_back(idx_cari);
  }

  if (is_free(idx_cari)==true)
  {
    _alpha_free.push_back(idx_cari);
  }


}

vector<bool> Tmy_list_alpha::is_alpha_sv(int idx)
{
  vector<bool> tmp;
  tmp.push_back((_alpha[idx]>=_lb) and (_alpha[idx]<=_ub) and (_alpha[idx]!=0.0));
  tmp.push_back((_alpha[idx]>_lb) and (_alpha[idx]<_ub) and (_alpha[idx]!=0.0));
  tmp.push_back((_alpha[idx]==_ub));
  tmp.push_back((_alpha[idx]==_lb));
  return tmp;
}

bool Tmy_list_alpha::is_lower_bound(int idx)
{
  return (_alpha_status[idx]==0);
}

bool Tmy_list_alpha::is_upper_bound(int idx)
{
  return (_alpha_status[idx]==1);
}

bool Tmy_list_alpha::is_free(int idx)
{
  return (_alpha_status[idx]==2);
}

vector<Tmy_double> Tmy_list_alpha::calculateBoundaries(int i,int j)
{
  Tmy_double t      = _alpha[i]+_alpha[j];
  Tmy_double diff   = t-_ub;
  Tmy_double diff1  = t+_lb;  
  vector<Tmy_double> hasil = {_lb,_ub};
  if(((_alpha[i]<=_ub) and (_alpha[i]>=_lb)) and ((_alpha[j]<=_ub) and (_alpha[j]>=_lb))){
    hasil = {max(diff,_lb),min(_ub,diff1)}; 
  }
  return hasil;
}

vector<Tmy_double> Tmy_list_alpha::limit_alpha(Tmy_double alpha_a,Tmy_double alpha_b,Tmy_double Low,Tmy_double High, int flag)
{
  vector<Tmy_double> hasil={alpha_a,alpha_b};
  if (alpha_a>High){
     if(flag==1)
     {
        Tmy_double s=alpha_a-High;
        hasil[1]= alpha_b+s;        
     }
     hasil[0]=High; 
  }else{
     if (alpha_a<Low){
      if(flag==1)
      {
        Tmy_double s=alpha_a-Low;
        hasil[1]= alpha_b+s;        
      }
      hasil[0]=Low; 
     } 
  }
  return hasil;
}

vector<Tmy_double> Tmy_list_alpha::calculateNewAlpha(int i,int j,Tmy_double delta,Tmy_double Low,Tmy_double High)
{
  Tmy_double alpha_a_new = _alpha[i]+delta;
  vector<Tmy_double> tmp = limit_alpha(alpha_a_new,0,Low,High,0);
  alpha_a_new = tmp[0];
  Tmy_double alpha_b_new = _alpha[j]+(_alpha[i]-alpha_a_new);
  tmp = limit_alpha(alpha_b_new,alpha_a_new,_lb,_ub,1);
  alpha_b_new = tmp[0];
  alpha_a_new = tmp[1];
  return {_alpha[i],_alpha[j],alpha_a_new,alpha_b_new};
}

Treturn_is_pass Tmy_list_alpha::is_pass(int i,int j,Tmy_double delta)
{
  Treturn_is_pass tmp;

  tmp.is_pass = false;
  tmp.alpha_i = _alpha[i];
  tmp.alpha_j = _alpha[j];
  tmp.new_alpha_i = _alpha[i];
  tmp.new_alpha_j = _alpha[j];
  
  if(i==j)
  {
    return tmp;
  }else{
    vector<Tmy_double> hsl=calculateBoundaries(i,j);
    Tmy_double Low=hsl[0],High=hsl[1];
    //cout <<"Low "<<Low<<" High "<<High<<endl;
    if(Low==High){       
       return tmp;
    }else{
       vector<Tmy_double> hsl=calculateNewAlpha(i,j,delta,Low,High);
       Tmy_double alpha_a_old=hsl[0],alpha_b_old=hsl[1],alpha_a_new=hsl[2],alpha_b_new=hsl[3];       
       double diff = alpha_a_new-alpha_a_old;       
       //abs(diff)<10e-5
       if(diff==0)
       {        
        return tmp;
       }else{
          tmp.is_pass = true;
          tmp.alpha_i = alpha_a_old;
          tmp.alpha_j = alpha_b_old;
          tmp.new_alpha_i = alpha_a_new;
          tmp.new_alpha_j = alpha_b_new;
          // cout<<"alpha_a_new : "<<alpha_a_new<<" alpha_a_old : "<<alpha_a_old<<endl;
          // cout<<"alpha_b_new : "<<alpha_b_new<<" alpha_b_old : "<<alpha_b_old<<endl;        
          return tmp;
       }
    }
  }  
}

Tmy_double Tmy_list_alpha::get_alpha(int idx)
{
  return _alpha[idx];  
}

vector<int> Tmy_list_alpha::get_list_lb_ub(int flag)
{
  if(flag==0){
     return _alpha_not_lb;
  }else{
    if(flag==1){
       return _alpha_not_ub;
    }else{
     if(flag==2){
       return _alpha_free;
     } 
    }  
  }
}

void Tmy_list_alpha::mv_lb_ub(int idx,int flag1)
{
   if(flag1==0)
   {
      for(int i=0;i<_alpha_not_lb.size();i++){
          if(_alpha_not_lb[i]==idx){
             _alpha_not_lb.erase(_alpha_not_lb.begin()+i);              
             _alpha_not_lb.insert(_alpha_not_lb.begin(),idx);
             break;
         }
      }    
    }
   else{
     if(flag1==1)
     {
         for(int i=0;i<_alpha_not_ub.size();i++){
             if(_alpha_not_ub[i]==idx){
             _alpha_not_ub.erase(_alpha_not_ub.begin()+i);
             _alpha_not_ub.insert(_alpha_not_ub.begin(),idx);   
             break;
            } 
         }
     }
   }
}

map<int,Tmy_double> Tmy_list_alpha::get_list_alpha_sv()
{
  return _alpha_sv;
}

Treturn_alpha_stat Tmy_list_alpha::get_stat()
{
  Treturn_alpha_stat tmp_stat;
  tmp_stat.jml_alpha = _jml_alpha;
  tmp_stat.n_all_sv  = _n_all_sv;
  tmp_stat.n_sv = _n_sv;
  tmp_stat.jml_alpha_n_sv = _jml_alpha_n_sv;
  return tmp_stat;
}

Tmy_double Tmy_list_alpha::get_ub()
{
  return _ub;
}

void Tmy_list_alpha::swap_index(int i, int j)
{
  swap(_alpha_status[i],_alpha_status[j]);
  swap(_alpha[i],_alpha[j]);

  update_alpha_sv(i);
  update_alpha_status(i);
  update_lb_ub(i);

  update_alpha_sv(j);
  update_alpha_status(j);
  update_lb_ub(j);


} 

