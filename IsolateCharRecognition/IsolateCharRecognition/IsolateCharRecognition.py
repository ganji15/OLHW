import time
import cPickle
import numpy
import MpfDecoder as mpf
from sklearn import neighbors
from Res import *
from Utils import *

PCA_dim = 300
LDA_dim = 150

def train(train_dir):
    
    files = get_files_of_dir(train_dir)
    #online_pca = run_online_pca(files, Feature_dim)
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    
    #online_lda = run_online_lda(get_files_of_dir(train_charac_dirs, '.txt'), PCA_dim, online_pca)
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    
    increase_mean = run_incrase_mean(files, LDA_dim, online_pca, PCA_dim, online_lda, LDA_dim)
    

def test(test_dir, is_log = True):
    
    print 'start load classifier...'
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    increase_mean = cPickle.load(open(knn_model_path, 'r'))

    mean_data = increase_mean.get_data()
    #mean_labels = increase_mean.get_labels()
    mean_labels = increase_mean.get_indexs()

    knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
    knn.fit(mean_data, mean_labels) 
    print 'success to load classifier.'
    
    err_count = 0
    total_count = 0

    files = get_files_of_dir(test_dir)
    for f in files:
        print f
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
            
        if is_log:
            log_file = f + '_res.txt'
            
            print 'error rate: %.2f%%'%(err_i * 100.0/ count_i)
            #print 'write result %s'%log_file
            #mpf.log_preds(labels, preds, log_file)
    
    dict = {'pca' : PCA_dim, 'lda' : LDA_dim,\
            'moxbox' : use_bc_metric}
    f = open(get_test_res_path(dict), 'w')
    f.write('err: %d, total: %d, err_rate: %.2f%%'\
            %(err_count, total_count, err_count * 100.0 / total_count))
    f.close()
    print ('err: %d, total: %d, err_rate: %.2f%%'\
            %(err_count, total_count, err_count * 100.0 / total_count))


if __name__ == '__main__':
    main(None, test)
    
    