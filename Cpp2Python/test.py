import MpfDecoder as m
import numpy
import struct

a = numpy.arange(0, 10, dtype = 'uint16')
b = numpy.arange(10, 20, dtype = 'uint16')
m.log_preds(a, b, 'D:\\test2.txt')
