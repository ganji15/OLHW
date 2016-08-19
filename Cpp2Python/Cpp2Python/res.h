#pragma once

typedef struct {
	unsigned char u_char_val;
	short short_val;
	float float_val;
}data_value;

typedef struct {
	unsigned char val[4] = { 0 };
}character_code;

// immutable
extern int size_of_header;
extern char format_code[8];
extern char *illustrations;
extern char code_type[20];
extern short code_length;
extern char data_type[20];
extern int sample_num;
extern int dimensionality;
extern character_code* labels;
extern data_value** values;
extern bool is_inited;
// mutable