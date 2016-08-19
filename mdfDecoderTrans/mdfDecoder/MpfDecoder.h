#include <string>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

typedef struct {
	unsigned char u_char_val;
	short short_val;
	float float_val;
}data_value;

typedef struct {
	char val[4] = { 0 };
}character_code;

class MpfDecoder
{
	public:
	MpfDecoder(const char* filename);
	MpfDecoder();
	~MpfDecoder();

	void loadFromFile(const char* filename, bool is_show = false);
	void loadFromFile_C(const char* filename, bool is_show = false);
	void writeToFile(const char* filename);
	void writeToFile2(const char* filename);
	void clear_buffer();
	data_value value_at(int i, int j);
	character_code label_at(int i);
	void write_value(data_value* value, character_code label, string save_dir);

	// immutable
	int size_of_header;
	char format_code[8];
	char *illustrations;
	char code_type[20];
	short code_length;
	char data_type[20];
	int sample_num;
	int dimensionality;
	character_code* labels;
	data_value** values;
	bool is_inited;
	// mutable

	private:
	void write_data_value(int i, int j, std::ofstream &f);
	void read_value_from_file(int i, int j, std::ifstream &f);
};
