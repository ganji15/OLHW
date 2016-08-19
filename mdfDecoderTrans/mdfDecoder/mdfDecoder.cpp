// mdfDecoder.cpp : 定义控制台应用程序的入口点。
//

#include "MpfDecoder.h"
#include <stdio.h>

MpfDecoder::MpfDecoder() {
	illustrations = NULL;
	labels = NULL;
	values = NULL;
	is_inited = false;
}

MpfDecoder::MpfDecoder(const char *filename)
{
	illustrations = NULL;
	labels = NULL;
	values = NULL;
	is_inited = false;

	loadFromFile(filename);
}


MpfDecoder::~MpfDecoder()
{
	clear_buffer();
}

void MpfDecoder::loadFromFile(const char* filename, bool is_show) {
	ifstream fin(filename, ios::binary);
	if (fin.is_open()) {
		clear_buffer();

		fin.read((char*)&size_of_header, sizeof(__int32));

		fin.read(format_code, sizeof(char) * 8);

		illustrations = new char[size_of_header - 62];
		fin.read(illustrations, sizeof(char) * (size_of_header - 62));

		fin.read(code_type, sizeof(char) * 20);

		fin.read((char*)&code_length, sizeof(__int16));

		fin.read(data_type, sizeof(char) * 20);

		fin.read((char*)&sample_num, sizeof(__int32));

		fin.read((char*)&dimensionality, sizeof(__int32));

		labels = new character_code[sample_num];
		values = new data_value*[sample_num];

		for (__int32 i = 0; i < sample_num; ++i) {
			fin.read((char*)labels[i].val, code_length);

			values[i] = new data_value[dimensionality];
			for (__int32 j = 0; j < dimensionality; ++j) {
				read_value_from_file(i, j, fin);
			}
		}
		is_inited = true;

		if (is_show) {
			cout << "size_of_header : " << size_of_header << endl;
			cout << "format_code : " << format_code << endl;
			cout << "illustration : " << illustrations << endl;
			cout << "code_type : " << code_type << endl;
			cout << "code_length : " << code_length << endl;
			cout << "data_type : " << data_type << endl;
			cout << "sample_num : " << sample_num << endl;
			cout << "dimensionality : " << dimensionality << endl;

			for (int i = 0; i < 5; ++i) {
				cout << labels[i].val << " ";
			}
			cout << endl;
		}
	}
	else {
		cout << "cann't open " << filename << endl;
		is_inited = false;
	}

	fin.close();
}

void MpfDecoder::loadFromFile_C(const char* filename, bool is_show) {
	FILE * fp = fopen(filename, "rb");
	if (fp == NULL) {
		cout << "cannot open" << filename << endl;
		return;
	}
	clear_buffer();

	fread(&size_of_header, sizeof(int), 1, fp);
	//fin.read((char*)&size_of_header, sizeof(__int32));
	fread(format_code, sizeof(char), 8, fp);
	//fin.read(format_code, sizeof(char) * 8);

	illustrations = new char[size_of_header - 62];
	fread(illustrations, sizeof(char), (size_of_header - 62), fp);
	//fin.read(illustrations, sizeof(char) * (size_of_header - 62));

	fread(code_type, sizeof(char), 20, fp);
	//fin.read(code_type, sizeof(char) * 20);

	fread(&code_length, sizeof(__int16), 1, fp);
	//fin.read((char*)&code_length, sizeof(__int16));

	fread(data_type, sizeof(char), 20, fp);
	//fin.read(data_type, sizeof(char) * 20);

	fread(&sample_num, sizeof(__int32), 1, fp);
	//fin.read((char*)&sample_num, sizeof(__int32));

	fread(&dimensionality, sizeof(__int32), 1, fp);
	//fin.read((char*)&dimensionality, sizeof(__int32));

	labels = new character_code[sample_num];
	values = new data_value*[sample_num];

	for (__int32 i = 0; i < sample_num; ++i) {
		fread(labels[i].val, sizeof(char), code_length, fp);
		//fin.read((char*)labels[i].val, code_length);

		values[i] = new data_value[dimensionality];
		for (__int32 j = 0; j < dimensionality; ++j) {
			//read_value_from_file(i, j, fin);
			fread(&values[i][j].u_char_val, sizeof(unsigned char), 1, fp);
		}
	}
	is_inited = true;
	
	fclose(fp);
}

void MpfDecoder::writeToFile(const char* filename) {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
		return;
	}
	ofstream fout(filename, 'w');
	if (fout.is_open()) {
		cout << "[begin] start write to file: " << filename << endl;
		fout << "size_of_header : " << size_of_header << endl;
		fout << "format_code : " << format_code << endl;
		fout << "illustration : " << illustrations << endl;
		fout << "code_type : " << code_type << endl;
		fout << "code_length : " << code_length << endl;
		fout << "data_type : " << data_type << endl;
		fout << "sample_num : " << sample_num << endl;
		fout << "dimensionality : " << dimensionality << endl;

		for (__int32 i = 0; i < sample_num; ++i) {

			for (int k = 0; k < code_length; ++k) {
				fout << (unsigned char)labels[i].val[k];
			}
			fout << " : ";

			for (__int32 j = 0; j < dimensionality; ++j) {
				write_data_value(i, j, fout);
			}
			fout << endl;
		}
	}
	else {
		cout << "cann't open " << filename << endl;
	}

	fout.close();
	cout << "[done] already write to file: " << filename << endl;
}

void MpfDecoder::writeToFile2(const char* filename) {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
		return;
	}
	ofstream fout(filename, 'w');
	if (fout.is_open()) {
		for (__int32 i = 0; i < sample_num; ++i) {
			/*
			for (int k = 0; k < code_length; ++k) {
				fout << (unsigned char)labels[i].val[k];
			}
			fout << " : ";
			*/

			for (__int32 j = 0; j < dimensionality; ++j) {
				write_data_value(i, j, fout);
			}
			fout << endl;
		}
	}
	else {
		cout << "cann't open " << filename << endl;
	}

	fout.close();
	cout << "[done] already write to file: " << filename << endl;
}

void MpfDecoder::clear_buffer() {
	if (illustrations != NULL)
		delete[] illustrations;
	if (labels != NULL)
		delete[] labels;
	if (values != NULL) {
		for (__int32 i = 0; i < sample_num; ++i) {
			delete[] values[i];
		}
		delete[] values;
	}

	is_inited = false;
}


void MpfDecoder::write_data_value(int i, int j, std::ofstream &f) {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
		return;
	}

	f << (int)values[i][j].u_char_val;
	/*
	if (strcmp(data_type, "unsigned char") == 0) {
		unsigned int val = bts2ui(&values[i][j].u_char_val);
		f << val;
	}
	else if (strcmp(data_type, "short") == 0) {
		f << values[i][j].short_val;
	}
	else if (strcmp(data_type, "float") == 0) {
		f << values[i][j].float_val;
	}
	*/

	f << " ";
}

void MpfDecoder::read_value_from_file(int i, int j, std::ifstream &f) {
	f.read((char*)&values[i][j].u_char_val, sizeof(unsigned char));
	/*
	if (strcmp(data_type, "unsigned char") == 0) {
		f.read((char*)&values[i][j].u_char_val, sizeof(unsigned char));
	}
	else if (strcmp(data_type, "short") == 0) {
		f.read((char*)&values[i][j].short_val, sizeof(short));
	}
	else if (strcmp(data_type, "float") == 0) {
		f.read((char*)&values[i][j].float_val, sizeof(float));
	}*/
}

data_value MpfDecoder::value_at(int i, int j) {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
	}

	if (i < 0 || i >= sample_num) {
		cout << "!Error!=>[MpfDecoder::value_at] out of bindary ! values["
			<< i << "][" << j << "]" << endl;
	}

	if (!is_inited || i < 0 || i >= sample_num) {
		throw - 1;
	}

	return values[i][j];
}

character_code MpfDecoder::label_at(int i) {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
	}

	if (i < 0 || i >= sample_num) {
		cout << "!Error!=>[MpfDecoder::label_at] out of bindary! labels["
			<< i << "]" << endl;
	}

	if (!is_inited || i < 0 || i >= sample_num) {
		throw - 1;
	}

	return labels[i];
}

void MpfDecoder::write_value(data_value* value, character_code label, string save_dir) {
	
	string filename = save_dir + string(label.val) + ".txt";
	/*
	ofstream fout(filename, 'w+');
	if (fout.is_open()) {
		for (int i = 0; i < dimensionality - 1; ++i)
			fout << (int)value[i].u_char_val << " ";
		fout << (int)value[dimensionality - 1].u_char_val << endl;
	}
	fout.close();
	*/

	FILE * fp = fopen(filename.c_str(), "a");
	if (fp == NULL) {
		cout << "cannot open" << filename << endl;
		return;
	}

	for (int i = 0; i < dimensionality - 1; ++i) {
		int val = (int)value[i].u_char_val;
		fprintf(fp, "%d", val);
		fputc(' ', fp);
	}
	int val = (int)value[dimensionality - 1].u_char_val;
	fprintf(fp, "%d", val);
	fputc('\r', fp);

	fclose(fp);
}