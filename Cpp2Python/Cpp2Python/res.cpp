#include "res.h"

// immutable
int size_of_header = 0;
char format_code[8] = { 0 };
char *illustrations = 0;
char code_type[20] = { 0 };
short code_length = 0;
char data_type[20] = { 0 };
int sample_num = 0;
int dimensionality = 0;
character_code* labels = 0;
data_value** values = 0;
bool is_inited = false;
// mutable