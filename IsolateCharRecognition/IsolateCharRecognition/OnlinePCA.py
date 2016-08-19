import numpy
from OnlineCov import OnlineCov
from Res import floatX

class OnlinePCA(OnlineCov):

    def __init__(self, dim):
        super(OnlinePCA, self).__init__(dim)
        self.eigVecs = numpy.zeros([dim, dim], dtype = floatX)


    def update_projection(self):
        cov = self.get_cov()
        eigVals, eigVecs = numpy.linalg.eig(cov)
        eigIdxs = numpy.argsort(eigVals.real)[::-1]
        self.eigVecs = numpy.asarray(eigVecs.real[:, eigIdxs], dtype = floatX)
        self.clear_buffer()

    def get_project_cov(self, new_dim):
        return self.eigVecs[:, 0 : new_dim]

    def get_mean(self):
        return numpy.asarray(self.mean, dtype = floatX)

    def project(self, values, new_dim, whiten = True, axes = 1):
        if not axes:
            values = values.transpose()

        cov = self.get_project_cov(new_dim)
        if whiten:
            white_values = numpy.dot(values - self.mean, cov)
        else:
            white_values = numpy.dot(values, cov)

        return white_values


    def clear_buffer(self):
        self.cov = []


'''
from OnlineCov import GpuOnlineCov

class GpuOnlinePCA(GpuOnlineCov):

    def __init__(self, dim):
        super(GpuOnlinePCA, self).__init__(dim)
        self.eigVecs = numpy.zeros([dim, dim], dtype = floatX)


    def update_projection(self):
        cov = self.get_cov()
        eigVals, eigVecs = numpy.linalg.eig(cov)
        eigIdxs = numpy.argsort(eigVals.real)[::-1]
        self.eigVecs = eigVecs[:, eigIdxs]
        

    def get_project_cov(self, new_dim):
        return self.eigVecs[:, 0 : new_dim]


    def project(self, values, new_dim, axes = 1):
        if not axes:
            values = values.transpose()

        cov = self.get_project_cov(new_dim)
        white_values = numpy.dot(values - self.mean, cov)
        return white_values
'''