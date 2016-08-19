import time, cPickle
from Res import *
from Utils import *
import numpy
from sklearn import neighbors

class LVQ(object):
   
    def __init__(self, sigma = 1e-1, p = 1e-1, change_rate = 1e-2, prototype_init = None):        
        if prototype_init is None:
            prototype_init = cPickle.load( open(knn_model_path, 'r'))
        
        self.mean_data = prototype_init.get_data()
        self.mean_idx = prototype_init.get_indexs()

        self.sigma = sigma
        self.p = p
        self.change_rate = change_rate


    def update_means(self, sample, target):
        dists =  numpy.sum((self.mean_data - sample)**2, 1)
        mk_idx = target
        mr_idx = numpy.argmin(dists)
        if mr_idx == target:
            return 0

        '''
        top2idx = numpy.argsort(dists)[:2]

        mk_idx = target
        mr_idx = top2idx[1]
        if self.mean_labels[ top2idx[0]] == target:
            return 0
        else:
            mr_idx = top2idx[0]
        '''
    
        mk = self.mean_data[mk_idx , :]
        mr = self.mean_data[mr_idx, :]

        delta_k = numpy.sum( numpy.square(sample - mk)) - numpy.sum( numpy.square(sample - mr))
        exp_delta_k = numpy.exp( -self.sigma * delta_k)
        mk_delta = - 2 * self.sigma * exp_delta_k * (sample - mk) / \
                        (1 + exp_delta_k) ** 2
        mr_delta = 2 * self.sigma * exp_delta_k * (sample - mr) / \
                        (1 + exp_delta_k) ** 2
        
        mk = mk - self.p * mk_delta
        mr = mr - self.p * mr_delta

        self.mean_data[mk_idx , :] = mk
        self.mean_data[mr_idx, :] = mr

        loss = 1 / (1 + exp_delta_k)

        return loss

    def update_vargin(self):
        self.p = self.p * (1 - self.change_rate)
        self.sigma = self.sigma * (1 + self.change_rate)

    def get_means(self):
        return self.mean_data

    def get_indexs(self):
        return self.mean_idx


def calc_train_err(test_dir , mean_data, mean_labels):
    print 'start load classifier...'
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    increase_mean = cPickle.load(open(knn_model_path, 'r'))

    knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
    knn.fit(mean_data, mean_labels) 
    print 'success to load classifier.'
    
    err_count = 0
    total_count = 0

    start_time = time.time()

    files = get_files_of_dir(test_dir)
    for f in files:
        mpf.loadFromFile(f)
        if not mpf._is_inited():
            continue

        values = numpy.zeros([mpf._sample_num(), mpf._dim()], dtype = floatX)
        mpf.get_values(values, value_normalized)
        if use_bc_metric:
            values = numpy.sqrt(values)
        
        values = online_pca.project(values, PCA_dim, PCA_whiten)
        values = online_lda.project(values, LDA_dim)

        labels = numpy.arange(0, mpf._sample_num(), dtype = 'uint16')
        mpf.get_labels(labels)
        labels = labels_to_index(labels, increase_mean.get_label2idx_dict())
        
        preds = knn.predict(values)

        err_i = sum(preds != labels)
        count_i = labels.size
        err_count += err_i
        total_count += count_i
        
        print f
        print 'error rate: %.2f%%'%(err_i * 100.0/ count_i)

    print ('time: %.2f err: %d, total: %d, err_rate: %.2f%%'\
            %(time.time() - start_time, err_count, total_count, err_count * 100.0 / total_count))


def run_lvq(train_dir = train_dir):
    PCA_dim = 300
    LDA_dim = 150
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    increase_mean = cPickle.load( open(knn_model_path, 'r'))
    data_bias = -online_pca.get_mean()

    trans = numpy.dot(online_pca.get_project_cov(PCA_dim), online_lda.get_project_cov(LDA_dim))
    values, idxs = load_all_data(train_dir, batch_size = 1,trans = trans, label2idx_dict = increase_mean.get_label2idx_dict(), bias = data_bias)
  
    lvq = LVQ()
    total_num = len(idxs)
    ten_percent_count = total_num / 10

    num_epoch = 100
    for epoch in xrange(num_epoch):
        calc_train_err(train_dir, lvq.get_means(), lvq.get_indexs())

        indices = range(0, len(idxs))
        numpy.random.shuffle(indices)

        start_time = time.time()
        loss = 0
        counter = 1
        for i in indices:
            loss += lvq.update_means(values[i, :], idxs[i])
            counter += 1
            if counter % ten_percent_count == 0:
                print 'percent: %.2f%%'%(counter * 1.0 / total_num * 100)

        
        loss = loss / len(idxs)
        lvq.update_vargin()
        print 'epoch %d: cost %.3fs, loss %.6f'%(epoch, time.time() - start_time, loss)
        cPickle.dump(lvq, open('D:\\lvq.pkl', 'w'))

        

    cPickle.dump(lvq, open('D:\\lvq.pkl', w))

if __name__ == '__main__':
    run_lvq()