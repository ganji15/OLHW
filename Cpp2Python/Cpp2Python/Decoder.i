%module MpfDecoder
%{
#define SWIG_FILE_WITH_INIT
#include "MpfDecoder.h"
#include "res.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int DIM1, double* INPLACE_ARRAY1) {(int c, double* invec)}
%apply (unsigned short* INPLACE_ARRAY1, int DIM1) {(unsigned short* invec, int n)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* invec, int r, int c)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* invec, int r, int c)}
%apply (int DIM1, unsigned short* IN_ARRAY1) {(int len1, unsigned short *invec1), (int len2, unsigned short *invec2)}
%feature("python:cdefaultargs") loadFromFile;
%feature("python:cdefaultargs") get_values;

%include "MpfDecoder.h"

%pythonbegin %{
# This module provides wrappers to the library
import numpy
def label_to_charac(label):
	b = numpy.zeros([1, 2], 'uint8')
	b[0, 0] = (label & 0xFF00) >> 8
	b[0, 1] = (label & 0x00FF)
	return ("{0:c}{1:c}").format(b[0,0], b[0, 1])

def charac_to_label(charac_str):
    return (ord(charac_str[0]) << 8) + ord(charac_str[1])
%}