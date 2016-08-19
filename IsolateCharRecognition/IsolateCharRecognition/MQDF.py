import numpy
from Res import *

class QDF(object):

    def __init__(self, dim, rows = Label_num):
        self.dim = dim
        self.SwInv = numpy.zeros([dim, dim, rows], dtype = floatX)
        self.means = numpy.zeros([rows, dim], dtype = floatX)
        self.counts = numpy.arange(0, rows, dtype = 'int')
        self.prob_wi = numpy.arange(0, rows, dtype = floatX)
        self.log_trace = numpy.arange(0, rows, dtype = floatX)
        self.labels = numpy.arange(0, rows, dtype = 'uint16')
        self.label_idx_dict = {}
        self.total_read = 0
        

    def add_cluster(self, values, label, h2 = 0.0):
        mean_value = numpy.mean(values, 0).reshape(1, -1)
        Swi = numpy.dot(values.transpose(), values) -\
                    numpy.dot(mean_value.transpose(), mean_value)

        Swi = Swi + h2 * h2 * numpy.eye(self.dim)
        tr_i = numpy.trace(Swi)
        self.log_trace[self.total_read] = numpy.log(tr_i) 

        self.SwInv[:, :, self.total_read] = numpy.matrix(Swi + tr_i / self.dim ** 2 * numpy.eye(self.dim)).I
        self.means[self.total_read, :] = mean_value[0, :]
        self.counts[self.total_read] = values.shape[0]
        self.labels[self.total_read] = label
        self.label_idx_dict[label] = self.total_read
               
        self.total_read += 1


    def fit(self, h_param = 0.0):
        self.prob_wi = self.counts * 1.0 / sum(self.counts)


    def clear_buffer(self):
        self.counts = []


    def predict(self, data, candidates, axes = 1):
        if not axes:
            data = numpy.transpose(data)

        preds = numpy.arange(0, data.shape[0], dtype = 'uint16')
        for i in range(0, data.shape[0]):
            min_score = numpy.inf
            idx = 0
            
            for j in range(0, candidates.shape[1]):
                candidate_wi = self.label_idx_dict[ candidates[i, j]]
                score = self.cal_score(data[i, :], candidate_wi)
                if score < min_score:
                    min_score = score
                    idx = candidate_wi
            
            preds[i] = self.labels[idx]

        return preds


    def cal_score(self, sample, w_i):
        diff = sample - self.means[w_i, :]
        score = numpy.dot( numpy.dot(diff, self.SwInv[:, :, w_i]),
                            numpy.transpose(diff)) +\
                self.log_trace[w_i] -\
                2 * numpy.log(self.prob_wi[w_i])

        return score
        

class MQDF(object):
    
    def __init__(self, dim, rows = Label_num, K = 0, h2 = 0.0):
        self.dim = dim
        if not K:
            K = dim

        self.SwEigVecs = numpy.zeros([dim, K, rows], dtype = floatX)
        self.SwEigVals = numpy.zeros([rows, K], dtype = floatX)
        self.means = numpy.zeros([rows, dim], dtype = floatX)
        self.counts = numpy.arange(0, rows, dtype = 'int')
        self.prob_wi = numpy.arange(0, rows, dtype = floatX)
        self.labels = numpy.arange(0, rows, dtype = 'uint16')
        self.label_idx_dict = {}
        self.total_read = 0
        self.K = K
        self.h2 = h2
        

    def add_cluster(self, values, label):
        mean_value = numpy.mean(values, 0).reshape(1, -1)
        Swi = numpy.dot(values.transpose(), values) -\
                    numpy.dot(mean_value.transpose(), mean_value)
        if self.h2 < 1e-5:
            self.h2 = numpy.sqrt( numpy.trace(Swi)) * 2.0 / self.dim

        Swi = Swi + self.h2 * self.h2 * numpy.eye(self.dim)

        eigVals, eigVecs = numpy.linalg.eig(Swi)
        eigIdxs = numpy.argsort(eigVals.real)[::-1]
        eigIdxs = eigIdxs[:self.K]

        self.SwEigVecs[:, :, self.total_read] = eigVecs.real[:, eigIdxs]
        self.SwEigVals[self.total_read, :] = eigVals.real[eigIdxs]

        self.means[self.total_read, :] = mean_value[0, :]
        self.counts[self.total_read] = values.shape[0]
        self.labels[self.total_read] = label
        self.label_idx_dict[label] = self.total_read
               
        self.total_read += 1


    def fit(self, h_param = 0.0):
        self.prob_wi = self.counts * 1.0 / sum(self.counts)


    def cal_score(self, sample, w_i):
        diff = sample - self.means[w_i, :]
        h2 = self.h2 * 1.0
        new_sample = numpy.dot(diff.reshape(1, -1), self.SwEigVecs[:, :, w_i])
        new_sample = numpy.array(new_sample)
        part_1 = numpy.dot(diff, diff.transpose()) -\
                 numpy.sum( (1 - h2 * h2 / self.SwEigVals[w_i, :])\
                            * new_sample ** 2\
                            )

        part_1 = 1 / h2 ** 2 * part_1
        part_2 = 2 * (self.dim - self.K) * numpy.log(h2) +\
                    numpy.sum( numpy.log(self.SwEigVals[w_i, :]))
        part_3 = -2 * numpy.log(self.prob_wi[w_i])

        score = part_1 + part_2 + part_3
        return score


    def clear_buffer(self):
        self.counts = []


    def predict(self, data, candidates, axes = 1):
        if not axes:
            data = numpy.transpose(data)

        preds = numpy.arange(0, data.shape[0], dtype = 'uint16')
        for i in range(0, data.shape[0]):
            min_score = numpy.inf
            idx = 0
            
            for j in range(0, candidates.shape[1]):
                candidate_wi = self.label_idx_dict[ candidates[i, j]]
                score = self.cal_score(data[i, :], candidate_wi)
                if score < min_score:
                    min_score = score
                    idx = candidate_wi
            
            preds[i] = self.labels[idx]

        return preds

