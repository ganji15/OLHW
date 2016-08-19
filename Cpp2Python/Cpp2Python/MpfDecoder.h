#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdint.h>

using namespace std;

void loadFromFile(const char* filename, bool is_show = false);
void writeToFile(const char* filename);
void clear_buffer();
void get_values(float *invec, int r, int c, bool normalized = 0);
void row_values(int c, double *invec, int r);
void get_labels(unsigned short *invec, int n);
void log_preds(int len1, unsigned short *invec1, int len2, unsigned short * invec2, char * filename);

int _sample_num();
int _dim();
char* _data_type();
char* _code_type();
short _code_length();
bool _is_inited();
void uint16_to_string(unsigned short val, unsigned char res[2]);

void read_uint8(ifstream &fin);
void read_short(ifstream &fin);
void read_float(ifstream &fin);

void write_uint8(ofstream &fout);
void write_short(ofstream &fout);
void write_float(ofstream &fout);