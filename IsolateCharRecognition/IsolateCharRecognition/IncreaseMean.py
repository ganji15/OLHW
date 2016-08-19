import os, cPickle
import numpy
from sklearn.neighbors import NearestNeighbors
from Res import Label_num, floatX, dict_model_path, train_dir
from OnlineSupviseDict import OnlineSupviseDict

class IncreaseMean(OnlineSupviseDict):

    def __init__(self, rows = Label_num, cols = 512):
        super(IncreaseMean, self).__init__(rows)
        self.data = numpy.zeros([rows, cols], dtype = floatX)


    def add_sample(self, label, value):
        super(IncreaseMean, self).add_sample(label)
        idx = self.label_idx_dict[label]
        cur_label_num = self.label_per_count[idx]
        self.data[idx, :] = (cur_label_num - 1) * self.data[idx, :] / cur_label_num\
                            + value / cur_label_num
    
    def get_data(self):
        return self.data[0 : self.label_num, :]

    def get_label2idx_dict(self):
        return super(IncreaseMean, self).get_label2idx_dict()

    def get_indexs(self):
        labels = self.get_labels()
        label2idx_dict = self.get_label2idx_dict()
        num_labels = len(labels)
        indexs = numpy.arange(0, num_labels, dtype = 'int')
        for i in xrange(num_labels):
            indexs[i] = label2idx_dict[ labels[i]]
        return indexs



class WeakKPC(IncreaseMean):
    
    def __init__(self, rows = Label_num, cols = 512):
        super(WeakKPC, self).__init__(rows, cols)
        self.nbrs = None
        self.num_candidates = 0

    def fit(self, num_candidates):
        self.nbrs = NearestNeighbors(n_neighbors = num_candidates).fit(self.data)
        self.num_candidates = num_candidates
        self.clear_buffer()

    def predict_candidates(self, data):
        distances, indices = self.nbrs.kneighbors(data)
        candidates = numpy.zeros([data.shape[0], self.num_candidates], dtype = 'uint16')

        for i in range(0, data.shape[0]):
            for j in xrange(self.num_candidates):
                candidates[i, j] = self.idx_label_dict[ indices[i, j]]

        return candidates

    def hit_count(self, data, labels):
        distances, indices = self.nbrs.kneighbors(data)
        candidates = numpy.zeros([data.shape[0], self.num_candidates], dtype = 'uint16')
        hits = 0

        for i in range(0, data.shape[0]):
            i_hit = 0
            for j in xrange(self.num_candidates):
                candidates[i, j] = self.idx_label_dict[ indices[i, j]]
                if not i_hit and candidates[i, j] == labels[i]:
                    i_hit = 1
            hits += i_hit

        return hits    

    def clear_buffer(self):
        self.data = []