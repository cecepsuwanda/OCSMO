#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/shm.h>	 /* shared memory functions and structs. */
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <bits/stdc++.h>
using namespace std;

#ifndef Included_Twrite_file_H

#define Included_Twrite_file_H

class Twrite_file
{
private:
   string _nm_f;
   FILE *_file = NULL;
   struct stat _sb;
   void open_file();

public:
	Twrite_file();
	~Twrite_file();

	void setnm_f(string nm_f);
	void close_file();

    void write_file(vector<string> &data);
	void write_file(string data);

};

#endif