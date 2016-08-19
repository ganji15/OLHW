# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_Decoder', [dirname(__file__)])
        except ImportError:
            import _Decoder
            return _Decoder
        if fp is not None:
            try:
                _mod = imp.load_module('_Decoder', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _Decoder = swig_import_helper()
    del swig_import_helper
else:
    import _Decoder
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


class data_value(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, data_value, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, data_value, name)
    __repr__ = _swig_repr
    __swig_setmethods__["u_char_val"] = _Decoder.data_value_u_char_val_set
    __swig_getmethods__["u_char_val"] = _Decoder.data_value_u_char_val_get
    if _newclass:
        u_char_val = _swig_property(_Decoder.data_value_u_char_val_get, _Decoder.data_value_u_char_val_set)
    __swig_setmethods__["short_val"] = _Decoder.data_value_short_val_set
    __swig_getmethods__["short_val"] = _Decoder.data_value_short_val_get
    if _newclass:
        short_val = _swig_property(_Decoder.data_value_short_val_get, _Decoder.data_value_short_val_set)
    __swig_setmethods__["float_val"] = _Decoder.data_value_float_val_set
    __swig_getmethods__["float_val"] = _Decoder.data_value_float_val_get
    if _newclass:
        float_val = _swig_property(_Decoder.data_value_float_val_get, _Decoder.data_value_float_val_set)

    def __init__(self):
        this = _Decoder.new_data_value()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _Decoder.delete_data_value
    __del__ = lambda self: None
data_value_swigregister = _Decoder.data_value_swigregister
data_value_swigregister(data_value)

class character_code(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, character_code, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, character_code, name)
    __repr__ = _swig_repr
    __swig_setmethods__["val"] = _Decoder.character_code_val_set
    __swig_getmethods__["val"] = _Decoder.character_code_val_get
    if _newclass:
        val = _swig_property(_Decoder.character_code_val_get, _Decoder.character_code_val_set)

    def __init__(self):
        this = _Decoder.new_character_code()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _Decoder.delete_character_code
    __del__ = lambda self: None
character_code_swigregister = _Decoder.character_code_swigregister
character_code_swigregister(character_code)

class MpfDecoder(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MpfDecoder, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MpfDecoder, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _Decoder.new_MpfDecoder(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _Decoder.delete_MpfDecoder
    __del__ = lambda self: None

    def loadFromFile(self, *args):
        return _Decoder.MpfDecoder_loadFromFile(self, *args)

    def writeToFile(self, filename):
        return _Decoder.MpfDecoder_writeToFile(self, filename)

    def clear_buffer(self):
        return _Decoder.MpfDecoder_clear_buffer(self)

    def value_at(self, i, j):
        return _Decoder.MpfDecoder_value_at(self, i, j)

    def label_at(self, i):
        return _Decoder.MpfDecoder_label_at(self, i)
    __swig_getmethods__["size_of_header"] = _Decoder.MpfDecoder_size_of_header_get
    if _newclass:
        size_of_header = _swig_property(_Decoder.MpfDecoder_size_of_header_get)
    __swig_getmethods__["format_code"] = _Decoder.MpfDecoder_format_code_get
    if _newclass:
        format_code = _swig_property(_Decoder.MpfDecoder_format_code_get)
    __swig_getmethods__["illustrations"] = _Decoder.MpfDecoder_illustrations_get
    if _newclass:
        illustrations = _swig_property(_Decoder.MpfDecoder_illustrations_get)
    __swig_getmethods__["code_type"] = _Decoder.MpfDecoder_code_type_get
    if _newclass:
        code_type = _swig_property(_Decoder.MpfDecoder_code_type_get)
    __swig_getmethods__["code_length"] = _Decoder.MpfDecoder_code_length_get
    if _newclass:
        code_length = _swig_property(_Decoder.MpfDecoder_code_length_get)
    __swig_getmethods__["data_type"] = _Decoder.MpfDecoder_data_type_get
    if _newclass:
        data_type = _swig_property(_Decoder.MpfDecoder_data_type_get)
    __swig_getmethods__["sample_num"] = _Decoder.MpfDecoder_sample_num_get
    if _newclass:
        sample_num = _swig_property(_Decoder.MpfDecoder_sample_num_get)
    __swig_getmethods__["dimensionality"] = _Decoder.MpfDecoder_dimensionality_get
    if _newclass:
        dimensionality = _swig_property(_Decoder.MpfDecoder_dimensionality_get)
    __swig_getmethods__["labels"] = _Decoder.MpfDecoder_labels_get
    if _newclass:
        labels = _swig_property(_Decoder.MpfDecoder_labels_get)
    __swig_getmethods__["values"] = _Decoder.MpfDecoder_values_get
    if _newclass:
        values = _swig_property(_Decoder.MpfDecoder_values_get)
    __swig_getmethods__["is_inited"] = _Decoder.MpfDecoder_is_inited_get
    if _newclass:
        is_inited = _swig_property(_Decoder.MpfDecoder_is_inited_get)
MpfDecoder_swigregister = _Decoder.MpfDecoder_swigregister
MpfDecoder_swigregister(MpfDecoder)

# This file is compatible with both classic and new-style classes.

