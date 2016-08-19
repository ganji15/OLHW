import os, time
import cPickle
import numpy
import MpfDecoder as mpf
from Res import *
from Utils import get_files_of_dir, get_test_res_path,run_online_pca, run_online_lda, run_weak_kpc, run_qdf

PCA_dim = 300
LDA_dim = 150
num_candidates = 10
h2 = 5

def train(train_dir):
    
    files = get_files_of_dir(train_dir)
    #online_pca = run_online_pca(files, Feature_dim)
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    
    #online_lda = run_online_lda(get_files_of_dir(train_charac_dirs, '.txt'), PCA_dim, online_pca)
    online_lda = cPickle.load(open(lda_model_path, 'r'))

    run_weak_kpc(files, num_candidates, LDA_dim, online_pca, PCA_dim, online_lda, LDA_dim)

    files = get_files_of_dir(train_charac_dirs, '.txt')
    run_qdf(files, online_pca, PCA_dim, online_lda, LDA_dim, h2)


def test(test_dir, is_log = False):
    
    print 'start load classifier...'
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    print 'pca loaded.'
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    print 'lda loaded.'
    weak_kpc = cPickle.load(open(weak_kpc_model_path, 'r'))
    print 'weak_kpc loaded.'
    mqdf = cPickle.load(open(qdf_model_path, 'r'))
    print 'mqdf loaded.'
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
        mpf.get_values(values)
        if use_bc_metric:
            values = numpy.sqrt(values)
        values = online_pca.project(values, PCA_dim)
        values = online_lda.project(values, LDA_dim)

        labels = numpy.arange(0, mpf._sample_num(), dtype = 'uint16')
        mpf.get_labels(labels)

        candidates = weak_kpc.predict_candidates(values)
        preds = mqdf.predict(values, candidates)

        err_i = sum(preds != labels)
        count_i = labels.size
        err_count += err_i
        total_count += count_i
            
        if is_log:
            log_file = f + '_res.txt'
            print 'write result %s'%log_file
            print 'error rate: %.2f%%'%(err_i * 100.0/ count_i)
            mpf.log_preds(labels, preds, log_file)
    
    dict = {'pca' : PCA_dim, 'lda' : LDA_dim,\
            'qdf' : 1}
    f = open(get_test_res_path(dict), 'w')
    f.write('err: %d, total: %d, err_rate: %.2f%%'\
            %(err_count, total_count, err_count * 100.0 / total_count))
    f.close()
    print ('err: %d, total: %d, err_rate: %.2f%%'\
            %(err_count, total_count, err_count * 100.0 / total_count))


if __name__ == '__main__':
    print 'start...'
    #'''
    start_time = time.time()
    train( train_dir)
    end_time = time.time()
    mins = (end_time - start_time) / 60
    secs = (end_time - start_time) % 60
    print 'Train cost:%2dm %2ds\n'%(mins, secs)
    #'''

    start_time = time.time()
    test( test_dir, is_log = True)
    end_time = time.time()
    mins = (end_time - start_time) / 60
    secs = (end_time - start_time) % 60
    print 'Test cost:%2dm %2ds'%(mins, secs)
    #'''

    print 'end...'
    