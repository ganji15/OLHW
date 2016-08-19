import numpy
from Res import floatX

class OnlineCov(object):

    def __init__(self, dim = 512):
        self.dim = dim
        self.cov = numpy.zeros([dim, dim], dtype = floatX)
        self.mean = numpy.zeros([1, dim], dtype = floatX)
        self.total_num = 0


    def add_batch_data(self, values, axes = 0):
        if values.shape[1 - axes] != self.dim:
            print 'invalid data, except:(%d), input:(%d,%d)'\
                    %(self.dim, values.shape[0], values.shape[1])
            return

        if axes:
            values = values.transpose()

        new_num = values.shape[0]
        self.cov += numpy.dot(values.transpose(), values)
        self.mean = self.total_num * self.mean / (self.total_num + new_num) +\
                    sum(values, 0) / (self.total_num + new_num)
        self.total_num += new_num


    def add_sample(self, value, weight = 1.):
        if value.size != self.dim:
            print 'invalid data, except:(%d), input:(%d)'\
                    %(self.dim, value.size)
            return

        value = value.reshape(1, -1)
        self.cov += numpy.dot(value.transpose(), value)
        self.mean = self.total_num * self.mean / (self.total_num + 1) +\
                    value / (self.total_num + 1)
        self.total_num += 1


    def get_cov(self, shrinkage = 0.):
        cov = self.cov - numpy.dot(self.mean.transpose(), self.mean)
        shrik_cov = (1 - shrinkage) * cov + shrinkage * numpy.eye(self.dim, dtype = floatX)
        return shrik_cov

    def clear_buffer(self):
        self.cov = []
        self.mean = []

'''
import theano
import theano.tensor as T

class GpuOnlineCov(object):

    def __init__(self, dim = 512):
        self.dim = dim
        self.cov = theano.shared(numpy.zeros([dim, dim], dtype = theano.config.floatX), name = 'olcov')
        self.mean = numpy.zeros([1, dim], dtype = theano.config.floatX)
        self.total_num = 0

        self.x = T.fmatrix('x')
        self.x2 = T.fmatrix('x2')
        self.y = T.iscalar('y')
        
        self.cov_update = theano.function([self.x], self.cov, updates = [(self.cov, self.cov + T.dot(self.x.T, self.x))])

    def add_batch_data(self, values, axes = 0):
        if values.shape[1 - axes] != self.dim:
            print 'invalid data, except:(%d), input:(%d,%d)'\
                    %(self.dim, values.shape[0], values.shape[1])
            return

        if axes:
            values = values.transpose()
        
        self.cov_update(values)
        new_num = values.shape[0]
        self.mean = self.total_num * self.mean / (self.total_num + new_num) +\
                    sum(values, 0) / (self.total_num + new_num)
        self.total_num += 1


    def get_cov(self, shrinkage = 0.):
        cov = self.cov.eval() - numpy.dot(self.mean.transpose(), self.mean)
        shrik_cov = (1 - shrinkage) * cov + shrinkage * numpy.eye(self.dim, dtype = theano.config.floatX)
        return shrik_cov

    def clear_buffer(self):
        self.cov = []
        self.mean = []
'''