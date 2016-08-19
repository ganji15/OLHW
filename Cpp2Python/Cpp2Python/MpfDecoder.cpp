#include "MpfDecoder.h"
#include "res.h"

void read_uint8(ifstream &fin) {
	for (__int32 i = 0; i < sample_num; ++i) {
		fin.read((char*)labels[i].val, code_length);

		values[i] = new data_value[dimensionality];
		for (__int32 j = 0; j < dimensionality; ++j) {
			fin.read((char*)&values[i][j].u_char_val, sizeof(unsigned char));
		}
	}
}

void read_short(ifstream &fin) {
	for (__int32 i = 0; i < sample_num; ++i) {
		fin.read((char*)labels[i].val, code_length);

		values[i] = new data_value[dimensionality];
		for (__int32 j = 0; j < dimensionality; ++j) {
			fin.read((char*)&values[i][j].short_val, sizeof(short));
		}
	}
}

void read_float(ifstream &fin) {
	for (__int32 i = 0; i < sample_num; ++i) {
		fin.read((char*)labels[i].val, code_length);

		values[i] = new data_value[dimensionality];
		for (__int32 j = 0; j < dimensionality; ++j) {
			fin.read((char*)&values[i][j].float_val, sizeof(float));
		}
	}
}

void write_uint8(ofstream &fout) {
	for (__int32 i = 0; i < sample_num; ++i) {

		for (int k = 0; k < code_length; ++k) {
			fout << (unsigned char)labels[i].val[k];
		}
		fout << " : ";

		for (__int32 j = 0; j < dimensionality; ++j) {
			fout << (int)values[i][j].u_char_val << " ";
		}
		fout << endl;
	}
}

void write_short(ofstream &fout) {
	for (__int32 i = 0; i < sample_num; ++i) {

		for (int k = 0; k < code_length; ++k) {
			fout << (unsigned char)labels[i].val[k];
		}
		fout << " : ";

		for (__int32 j = 0; j < dimensionality; ++j) {
			fout << (int)values[i][j].short_val << " ";
		}
		fout << endl;
	}
}

void write_float(ofstream &fout) {
	for (__int32 i = 0; i < sample_num; ++i) {

		for (int k = 0; k < code_length; ++k) {
			fout << (unsigned char)labels[i].val[k];
		}
		fout << " : ";

		for (__int32 j = 0; j < dimensionality; ++j) {
			fout << values[i][j].float_val << " ";
		}
		fout << endl;
	}
}

void loadFromFile(const char* filename, bool is_show) {
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

		if (strcmp(data_type, "unsigned char") == 0) {
			read_uint8(fin);
		}
		else if (strcmp(data_type, "short") == 0) {
			read_short(fin);
		}
		else if (strcmp(data_type, "float") == 0) {
			read_float(fin);
		}
		else {
			cout << "unsupported data_type : " << data_type << endl;
			return;
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

void writeToFile(const char* filename) {
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

		if (strcmp(data_type, "unsigned char") == 0) {
			write_uint8(fout);
		}
		else if (strcmp(data_type, "short") == 0) {
			write_short(fout);
		}
		else if (strcmp(data_type, "float") == 0) {
			write_float(fout);
		}
		else {
			cout << "unsupported data_type : " << data_type << endl;
			return;
		}
	}
	else {
		cout << "cann't open " << filename << endl;
	}

	fout.close();
	cout << "[done] already write to file: " << filename << endl;
}

void clear_buffer() {
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

void get_values(float *invec, int r, int c, bool normalized) {
	if (!is_inited) {
		cout << "please load *.mpf file first" << endl;
		return;
	}
	
	float scale = 1.0;
	if (normalized)
		scale = 1.0 / 256.0;

	if (strcmp(data_type, "unsigned char") == 0) {
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				invec[i * c + j] = \
				static_cast<float>(values[i][j].u_char_val) * scale;
	}
	else if (strcmp(data_type, "short") == 0) {
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				invec[i * c + j] = \
				static_cast<float>(values[i][j].short_val) * scale;
	}
	else if (strcmp(data_type, "float") == 0) {
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				invec[i * c + j] = (values[i][j].float_val) * scale;
	}
	else {
		cout << "unsupported data_type: " << data_type << endl;
	}

}

void row_values(int c, double *invec, int r) {
	if (!is_inited) {
		cout << "please load *.mpf file first" << endl;
		return;
	}

	if (strcmp(data_type, "unsigned char") == 0) {
			for (int j = 0; j < c; ++j)
				invec[j] = \
				static_cast<double>(values[r][j].u_char_val);
	}
	else if (strcmp(data_type, "short") == 0) {
			for (int j = 0; j < c; ++j)
				invec[j] = \
				static_cast<double>(values[r][j].short_val);
	}
	else if (strcmp(data_type, "float") == 0) {
			for (int j = 0; j < c; ++j)
				invec[j] = (values[r][j].float_val);
	}
	else {
		cout << "unsupported data_type: " << data_type << endl;
	}
}

void get_labels(unsigned short *invec, int n) {
	if (!is_inited) {
		cout << "please load *.mpf file first" << endl;
		return;
	}

	for (int i = 0; i < n; ++i){
		invec[i] = ((unsigned short)labels[i].val[0] << 8) | (unsigned short)labels[i].val[1];
	}
}

int _sample_num() {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
	}

	return sample_num;
}

int _dim() {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
	}

	return dimensionality;
}

char* _data_type() {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
		return 0;
	}

	return data_type;
}

char* _code_type() {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
		return 0;
	}

	return code_type;
}

short _code_length() {
	if (!is_inited) {
		cout << "You need to load *.mpf file first!" << endl;
		return 0;
	}

	return code_length;
}

bool _is_inited() {
	return is_inited;
}

void log_preds(int len1, unsigned short *invec1, int len2, unsigned short * invec2, char * filename) {
	if (len1 != len2) {
		cout << "Array length not equal (" << len1 << "!= " << len2 << ");" << endl;
		return;
	}

	ofstream fout(filename, 'w');
	if (fout.is_open()) {
		int err_count = 0;
		int total_num = 0;
		unsigned char res[2];

		for (int i = 0; i < len1; ++i) {
			int err = 0;
			if (invec1[i] != invec2[i]){
				err_count++;
				err = 1;
			}

			total_num++;
			uint16_to_string(invec1[i], res);
			fout << (unsigned char)res[0] << (unsigned char)res[1];
			fout << " ";
			uint16_to_string(invec2[i], res);
			fout << (unsigned char)res[0] << (unsigned char)res[1];
			fout << " " << err << endl;
		}

		fout << "errors: " << err_count << endl;
		fout << "total: " << total_num << endl;
		fout << "err_rate: " << (err_count * 100.0 / total_num) << "%" << endl;
	}
	fout.close();
}

void uint16_to_string(unsigned short val, unsigned char res[2]){
	res[0] = (val & 0xFF00) >> 8;
	res[1] = (val & 0x00FF);
}