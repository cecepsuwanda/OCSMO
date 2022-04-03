#include "Twrite_file.h"


Twrite_file::Twrite_file()
{

}

Twrite_file::~Twrite_file()
{

}

void Twrite_file::setnm_f(string nm_f)
{
	_nm_f = nm_f;
	open_file();
}

void Twrite_file::open_file()
{
	_file = fopen(_nm_f.c_str(), "a+");

	// if (fstat(_file, &_sb) == -1)
	// {
	// 	perror(" couldn't get file size. \n");
	// }
}

void Twrite_file::close_file()
{
	if (!(_file == NULL))
	{
		fclose(_file);
		_file = NULL;
	}

}

void Twrite_file::write_file(string data)
{
	fputs((char *) data.c_str(), _file);
	fputc('\n', _file);
}

void Twrite_file::write_file(vector<string> &data)
{
	for (size_t i = 0; i < data.size(); ++i)
	{
		write_file(data[i]);
	}

	close_file();
}