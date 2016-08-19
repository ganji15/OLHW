import os, time
import cPickle
import numpy
import MpfDecoder as mpf
from OnlinePCA import OnlinePCA
from OnlineLDA import OnlineLDA
from IncreaseMean import WeakKPC
from MQDF import QDF
from sklearn import neighbors
from Res import *
from Utils import get_files_of_dir

Feature_dim = 512
PCA_dim = 300
LDA_dim = 150
num_candidates = 40
h2 = 10

def run_online_pca(files):
    online_pca = OnlinePCA(Feature_dim)

    for f in files:
        print '[OnlinePCA] : ' + f

        mpf.loadFromFile(f)
        if not mpf._is_inited():
            continue
            
        values = numpy.zeros([mpf._sample_num(), mpf._dim()], dtype = floatX)
        mpf.get_values(values)
        if use_bc_metric:
            values = numpy.sqrt(values)

        online_pca.add_batch_data(values)
    
    print '[OnlinePCA] : cal_projection' 
    online_pca.update_projection()

    cPickle.dump(online_pca, open(pca_model_path, 'w'))
    print 'save online_pca to %s'%(pca_model_path)
    return online_pca


def run_online_lda(files, online_pca):
    online_lda = OnlineLDA(PCA_dim)

    file_count = 1
    for f in files:
        print '[OnlineLDA] : %s%5d'%(f, file_count)
        file_count += 1

        cluster = numpy.loadtxt(f, dtype = floatX)
        if use_bc_metric:
            cluster = numpy.sqrt(cluster)

        cluster = online_pca.project(cluster, PCA_dim)
        online_lda.add_cluster(cluster)

    print '[OnlineLDA] : cal_projection' 
    online_lda.update_projection()

    cPickle.dump(online_lda, open(lda_model_path, 'w'))
    print 'save online_lda to %s'%(lda_model_path)

    return online_lda


def run_weak_kpc(files, online_pca, online_lda):
    weak_kpc = WeakKPC(Label_num, LDA_dim)

    for f in files:
        print '[weak_KPC] : ' + f

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

        for i in range(0, mpf._sample_num()):
            weak_kpc.add_sample(labels[i], values[i, :])

    weak_kpc.fit(num_candidates)
    
    cPickle.dump(weak_kpc, open(weak_kpc_model_path, 'w'))
    print 'save weak_kpc_model to %s'%(weak_kpc_model_path)


def run_mqdf(files, online_pca, online_lda):
    mqdf = QDF(LDA_dim, Label_num)

    file_count = 1
    for f in files:
        print '[QDF] : %s%5d'%(f, file_count)
        file_count += 1

        cluster = numpy.loadtxt(f, dtype = floatX)
        if use_bc_metric:
            cluster = numpy.sqrt(cluster)

        cluster = online_pca.project(cluster, PCA_dim)
        cluster = online_lda.project(cluster, LDA_dim)
        label = mpf.charac_to_label( os.path.basename(f))
        mqdf.add_cluster(cluster, label, h2 = h2)

    print '[QDF] : cal_projection' 
    mqdf.fit()

    cPickle.dump(mqdf, open(qdf_model_path, 'w'))
    print 'save mqdf to %s'%(qdf_model_path)


def train(train_dir):
    
    files = get_files_of_dir(train_dir)
    #online_pca = run_online_pca(files)
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    #online_lda = run_online_lda(get_files_of_dir(train_charac_dirs, '.txt'), online_pca)
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    #online_lda.update_projection(shrinkage = shrinkage)
    #cPickle.dump(online_lda, open(lda_model_path, 'w'))
    run_weak_kpc(files, online_pca, online_lda)


def test(test_dir, is_log = False):
    
    print 'start load classifier...'
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    print 'pca loaded.'
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    print 'lda loaded.'
    weak_kpc = cPickle.load(open(weak_kpc_model_path, 'r'))
    print 'weak_kpc loaded.'
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

        hits = weak_kpc.hit_count(values, labels)

        err_i = labels.size - hits
        count_i = labels.size
        err_count += err_i
        total_count += count_i
    
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
    