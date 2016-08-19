import numpy
from OnlineCov import OnlineCov
from Res import Label_num, floatX

class OnlineLDA(object):

    def __init__(self, dim, rows = Label_num):
        self.dim = dim
        self.Sw = numpy.zeros([dim, dim], dtype = floatX)
        self.means = numpy.zeros([rows, dim], dtype = floatX)
        self.counts = numpy.arange(0, rows, dtype = 'int')
        self.total_read = 0
        self.eigVecs = numpy.matrix([])


    def add_cluster(self, values):
        mean_value = numpy.mean(values, 0).reshape(1, -1)
        self.Sw += numpy.dot(values.transpose(), values) -\
                    numpy.dot(mean_value.transpose(), mean_value)
        self.means[self.total_read, :] = mean_value[0, :]
        self.counts[self.total_read] = values.shape[0]
        self.total_read += 1


    def update_projection(self, is_centered = True, shrinkage = 0.):
        if not is_centered:
            mean = numpy.sum(self.counts.reshape(self.total_read, 1)\
                    * self.means, 0) / numpy.sum(self.counts)
        else:
            mean = numpy.zeros([1, self.dim], dtype = floatX)
        
        Sb = numpy.zeros([self.dim, self.dim], dtype = floatX)
        for i in range(0, self.total_read):
            diff = self.means[i, :] - mean
            diff = diff.reshape(1, -1)
            Sb += self.counts[i] *\
                    numpy.dot(diff.transpose(), diff)

        shrink_Sw = (1 - shrinkage) * self.Sw +\
                    shrinkage * numpy.eye(self.dim) * self.total_read
        
        #shrink_Sw = numpy.matrix(self.Sw)
        shrink_Sw = numpy.matrix(shrink_Sw)
        Sb = numpy.matrix(Sb)
        cov = (shrink_Sw.T * shrink_Sw).I * shrink_Sw.T * Sb
        eigVals, eigVecs = numpy.linalg.eig(cov)
        eigIdxs = numpy.argsort(eigVals.real)[::-1]
        self.eigVecs = numpy.asarray(eigVecs.real[:, eigIdxs], dtype = floatX)

        self.clear_buffer()


    def get_project_cov(self, new_dim):
        return self.eigVecs[:, 0 : new_dim]


    def project(self, values, new_dim, axes = 1):
        if not axes:
            values = values.transpose()

        values = numpy.matrix(values)
        cov = self.get_project_cov(new_dim)
        lda_values = values * cov
        return lda_values


    def clear_buffer(self):
        self.Sw = []
        self.means = []
        self.counts = []


class OLDirectLDA(OnlineLDA):

    def __init__(self, dim, rows = Label_num):
        super(OLDirectLDA, self).__init__(dim, rows)
        self.Sb = []

    def update_projection(self, d_dim = 0, u_dim = 0):
        mean = numpy.sum(self.counts.reshape(self.total_read, 1)\
                    * self.means, 0) / numpy.sum(self.counts)
        
        Sb = numpy.zeros([self.dim, self.dim], dtype = floatX)
        for i in range(0, self.total_read):
            diff = self.means[i, :] - mean
            diff = diff.reshape(1, -1)
            Sb += self.counts[i] *\
                    numpy.dot(diff.transpose(), diff)

        self.Sb = Sb
        eigVals, eigVecs = numpy.linalg.eig(Sb)
        print 'rank Sb: %d'%(numpy.linalg.matrix_rank(Sb))
        eigIdxs = numpy.argsort(eigVals.real)[::-1]
        new_Idxs = [idx for idx in eigIdxs if eigVals.real[idx] > 0]

        if d_dim > 0 and d_dim < len(new_Idxs):
            new_Idxs = new_Idxs[:d_dim]

        Y = numpy.matrix(eigVecs.real[:, new_Idxs])
        Db = eigVals.real[new_Idxs] * numpy.eye(len(new_Idxs))
        Db = numpy.matrix(Db)
        
        print 'Y rank: %d'%(len(new_Idxs))        

        Z = Y * numpy.sqrt(Db.I)
        Sw = numpy.matrix(self.Sw)
        Sw2 = Z.T * Sw * Z      

        eigVals, eigVecs = numpy.linalg.eig(Sw2)
        eigIdxs = numpy.argsort(eigVals.real)
        if u_dim > 0 and u_dim < eigIdxs.size:
            eigIdxs = eigIdxs[:u_dim]
        
        print 'Sw rank: %d'%(len(eigIdxs))          

        U = numpy.matrix(eigVecs.real[:, eigIdxs])
        Dw = numpy.matrix(eigVals.real[eigIdxs] * numpy.eye(len(eigIdxs)))
        
        A = Z * U * numpy.sqrt(Dw.I)
        print A.shape
        self.eigVecs = numpy.array(A)


'''
class OnlineLDA(OnlineSupviseDict):

    def __init__(self, dim, rows = Label_num):
        super(OnlineLDA, self).__init__(rows)
        self.class_cov = []
        self.dim = dim
        for i in range(0, rows):
            self.class_cov.append( OnlineCov(dim))

        self.eigVecs = numpy.zeros([dim, dim], dtype = floatX)
        self.total_mean = numpy.zeros([1, dim], dtype = floatX)


    def add_sample(self, label, value):
        super(OnlineLDA, self).add_sample(label)

        idx = self.label_idx_dict[label]
        self.class_cov[idx].add_sample(value)
        self.total_mean = (self.total_read - 1) * self.total_mean / self.total_read +\
                            value / self.total_read


    def update_projection(self):
        Sw = numpy.zeros([self.dim, self.dim], dtype = floatX)
        Sb = numpy.zeros([self.dim, self.dim], dtype = floatX)
        for i in range(0, self.label_num):
            Sw += self.class_cov[i].get_cov()
            diff = self.class_cov[i].mean - self.total_mean
            Sb += self.class_cov[i].total_num * numpy.dot(diff.transpose(), diff)
        cov = (Sw ** (-1)) * Sb
        eigVals, eigVecs = numpy.linalg.eig(cov)
        eigIdxs = numpy.argsort(eigVals.real)[::-1]
        self.eigVecs = eigVecs[:, eigIdxs]


    def get_project_cov(self, new_dim):
        return self.eigVecs[:, 0 : new_dim]


    def project(self, values, new_dim, axes = 1):
        if not axes:
            values = values.transpose()

        cov = self.get_project_cov(new_dim)
        lda_values = numpy.dot(values, cov)
        return lda_values
'''    