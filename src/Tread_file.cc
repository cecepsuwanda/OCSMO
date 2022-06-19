#include "Tread_file.h"

Tread_file::Tread_file()
{
}

Tread_file::~Tread_file()
{
  clear_data();
  // close_file();
  clear_index();
  clear_memory();
}

Tread_file::Tread_file(const Tread_file &t)
{
  cout << "Copy constructor called " << endl;
}

void Tread_file::set_shared_memory_on()
{
  _shared_memory = true;
}

void Tread_file::set_shared_memory_off()
{
  _shared_memory = false;
}

void Tread_file::setnm_f(string nm_f, const char *separator)
{
  _nm_f = nm_f;
  _separator = separator;
  if (open_file())
  {
    read_file();
  }
  else
  {
    cout << "Gagal buka " << _nm_f << " !!!" << endl;
  }
}

void Tread_file::setseparator(const char *separator)
{
  _separator = separator;
}

int Tread_file::get_jml_row()
{
  return _jml_row;
}

int Tread_file::get_jml_col()
{
  return _jml_col;
}

bool Tread_file::open_file(string mode)
{
  // _file = fopen(_nm_f.c_str(), (char *) mode.c_str());
  // return !(_file == NULL);
  return false;
}

bool Tread_file::open_file()
{

  _fd = open(_nm_f.c_str(), O_RDONLY, S_IRUSR | S_IWUSR); //, S_IRUSR | S_IWUSR

  if (fstat(_fd, &_sb) == -1)
  {
    cout << _nm_f << endl;
    perror(" couldn't get file size. \n");
  }
  else
  {
    memory_map_file();
  }

  // printf("file size is %ld\n",_sb.st_size);
  // close(_fd);
  return !(_fd == -1);
}

void Tread_file::memory_map_file()
{

  _file_in_memory = (char *)mmap(NULL, _sb.st_size, PROT_READ, MAP_PRIVATE, _fd, 0); // MAP_SHARED

  if (_file_in_memory == MAP_FAILED)
  {
    perror(" mmap error \n");
  }

  _posisi = 0;
  _idx_posisi = 0;
}

void Tread_file::close_file()
{
  munmap(_file_in_memory, _sb.st_size);
  _file_in_memory = NULL;
  _posisi = 0;
  _idx_posisi = 0;
  close(_fd);
}

vector<string> Tread_file::tokenizer(char *str, const char *separator)
{

  char *token;
  vector<string> vec;
  int len;

  len = strlen(str);
  if (str[len - 1] == '\n')
    str[len - 1] = 0;

  token = strtok(str, separator);
  while (token != 0)
  {
    vec.push_back(token);
    token = strtok(0, separator);
  }

  return vec;
}

void Tread_file::reset_file()
{
  _posisi = 0;
  _idx_posisi = 0;
  read_file();
}

void Tread_file::read_file()
{
  clear_data();
  char str[1000];

  strcpy(str, "");

  if (is_index and (_jml_index > 0))
  {
    _b_posisi = _idx_in_memory[_idx_posisi];
    int p_awal = _idx_in_memory[_idx_posisi];
    while ((p_awal < _sb.st_size) and (_file_in_memory[p_awal] != '\n'))
    {
      char tmp = _file_in_memory[p_awal];
      strncat(str, &tmp, 1);
      p_awal++;
    }
    char tmp = _file_in_memory[p_awal];
    strncat(str, &tmp, 1);
    _idx_posisi++;
  }
  else
  {
    _b_posisi = _posisi;
    while ((_posisi < _sb.st_size) and (_file_in_memory[_posisi] != '\n'))
    {
      char tmp = _file_in_memory[_posisi];
      strncat(str, &tmp, 1);
      _posisi++;
    }
    char tmp = _file_in_memory[_posisi];
    strncat(str, &tmp, 1);

    // if(_b_posisi==0)
    //    cout << str << endl;

    _posisi++;
  }
  // cout << str << endl;

  int len;

  len = strlen(str);
  if (str[len - 1] == '\n')
    str[len - 1] = 0;

  _data_str = str;
  _data = tokenizer(str, _separator);
  // cout << "Hasil Token : " << _data.size() <<endl;
  _idx_col = 0;
}

void Tread_file::clear_data()
{

  _data.clear();
  _data.shrink_to_fit();
}

void Tread_file::clear_index()
{

  _index.clear();
  _index.shrink_to_fit();
}

bool Tread_file::is_eof()
{

  if (is_index and (_jml_index > 0))
  {
    return (_idx_posisi > _jml_index);
  }
  else
  {
    return (_posisi > _sb.st_size);
  }
}

vector<string> Tread_file::get_record()
{

  return _data;
}

void Tread_file::next_col()
{
  _idx_col = _idx_col + 1;
}

bool Tread_file::is_end_col()
{

  return !(_idx_col < _data.size());
}

string Tread_file::get_col_val()
{

  return _data[_idx_col];
}

int Tread_file::get_idx_col()
{

  return _idx_col;
}

string Tread_file::get_col_val(int idx_col)
{
  try
  {
    return _data[idx_col];
  }
  catch (std::bad_alloc)
  {
    cout << _data.size() << " " << _idx_col << endl;
    return "-1";
  }
}

void Tread_file::next_record()
{
  read_file();
}

void Tread_file::index_on()
{
  is_index = true;
}

void Tread_file::index_off()
{
  is_index = false;
}

void Tread_file::add_index()
{
  _index.push_back(_b_posisi);
}

void Tread_file::save_to_memory()
{
  _jml_index = _index.size();
  _ukuran_index = (((_jml_index * sizeof(int)) / pagesize) + 1) * pagesize;

  _idx_in_memory = (int *)mmap(NULL, _ukuran_index, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  for (int i = 0; i < _jml_index; ++i)
  {
    _idx_in_memory[i] = _index[i];
  }
}

void Tread_file::clear_memory()
{
  if (_idx_in_memory != NULL)
  {
    munmap(_idx_in_memory, _ukuran_index);
    _idx_in_memory = NULL;
    _ukuran_index = 0;
    _jml_index = 0;
  }
}

void Tread_file::goto_rec(int idx)
{
  if (is_index and (_jml_index > 0))
  {
    _idx_posisi = idx;
  }
  else
  {
    _posisi = idx;
  }

  read_file();
}
