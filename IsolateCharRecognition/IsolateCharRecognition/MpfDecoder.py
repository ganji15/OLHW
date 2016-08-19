# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.


# This module provides wrappers to the library
import numpy
def label_to_charac(label):
	b = numpy.zeros([1, 2], 'uint8')
	b[0, 0] = (label & 0xFF00) >> 8
	b[0, 1] = (label & 0x00FF)
	return ("{0:c}{1:c}").format(b[0,0], b[0, 1])

def charac_to_label(charac_str):
    return (ord(charac_str[0]) << 8) + ord(charac_str[1])





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_MpfDecoder', [dirname(__file__)])
        except ImportError:
            import _MpfDecoder
            return _MpfDecoder
        if fp is not None:
            try:
                _mod = imp.load_module('_MpfDecoder', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _MpfDecoder = swig_import_helper()
    del swig_import_helper
else:
    import _MpfDecoder
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



def loadFromFile(*args):
    return _MpfDecoder.loadFromFile(*args)
loadFromFile = _MpfDecoder.loadFromFile

def writeToFile(filename):
    return _MpfDecoder.writeToFile(filename)
writeToFile = _MpfDecoder.writeToFile

def clear_buffer():
    return _MpfDecoder.clear_buffer()
clear_buffer = _MpfDecoder.clear_buffer

def get_values(*args):
    return _MpfDecoder.get_values(*args)
get_values = _MpfDecoder.get_values

def row_values(c, r):
    return _MpfDecoder.row_values(c, r)
row_values = _MpfDecoder.row_values

def get_labels(invec):
    return _MpfDecoder.get_labels(invec)
get_labels = _MpfDecoder.get_labels

def log_preds(len1, len2, filename):
    return _MpfDecoder.log_preds(len1, len2, filename)
log_preds = _MpfDecoder.log_preds

def _sample_num():
    return _MpfDecoder._sample_num()
_sample_num = _MpfDecoder._sample_num

def _dim():
    return _MpfDecoder._dim()
_dim = _MpfDecoder._dim

def _data_type():
    return _MpfDecoder._data_type()
_data_type = _MpfDecoder._data_type

def _code_type():
    return _MpfDecoder._code_type()
_code_type = _MpfDecoder._code_type

def _code_length():
    return _MpfDecoder._code_length()
_code_length = _MpfDecoder._code_length

def _is_inited():
    return _MpfDecoder._is_inited()
_is_inited = _MpfDecoder._is_inited

def uint16_to_string(val, res):
    return _MpfDecoder.uint16_to_string(val, res)
uint16_to_string = _MpfDecoder.uint16_to_string

def read_uint8(fin):
    return _MpfDecoder.read_uint8(fin)
read_uint8 = _MpfDecoder.read_uint8

def read_short(fin):
    return _MpfDecoder.read_short(fin)
read_short = _MpfDecoder.read_short

def read_float(fin):
    return _MpfDecoder.read_float(fin)
read_float = _MpfDecoder.read_float

def write_uint8(fout):
    return _MpfDecoder.write_uint8(fout)
write_uint8 = _MpfDecoder.write_uint8

def write_short(fout):
    return _MpfDecoder.write_short(fout)
write_short = _MpfDecoder.write_short

def write_float(fout):
    return _MpfDecoder.write_float(fout)
write_float = _MpfDecoder.write_float
# This file is compatible with both classic and new-style classes.

