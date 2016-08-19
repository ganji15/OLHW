import numpy
from Res import Label_num

class OnlineSupviseDict(object):

    def __init__(self, rows = Label_num):
        self.label_idx_dict = {}
        self.idx_label_dict = {}
        self.label_num = 0
        self.label_per_count = numpy.zeros([rows, 1], dtype = 'int')
        self.total_read = 0


    def add_sample(self, label):
        if not self.label_idx_dict.has_key(label):
            self.label_idx_dict[label] = self.label_num
            self.idx_label_dict[self.label_num] = label
            self.label_num += 1
        
        idx = self.label_idx_dict[label]
        self.label_per_count[idx] += 1
        self.total_read += 1


    def get_labels(self):
        items = self.idx_label_dict.items()
        labels = []
        for item in items:
            labels.append(item[1])
        return labels

    def get_label2idx_dict(self):
        return self.label_idx_dict

    def get_idx2label_dict(self):
        return self.idx_label_dict
