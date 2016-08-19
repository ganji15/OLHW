import os, cPickle, time
import numpy
import theano
import MpfDecoder as mpf
from OnlineSupviseDict import OnlineSupviseDict
from IncreaseMean import IncreaseMean, WeakKPC
from OnlinePCA import OnlinePCA
from OnlineLDA import OnlineLDA
from MQDF import QDF, MQDF
from Res import *

# MpfDecoder.log_preds(labels, preds, filename)
def log(filename, labels, preds):
    f = open(filename, 'w')
    errs = (labels != preds)
    for label, pred, err in zip(labels, preds, errs):
        label_str = mpf.label_to_charac(label)
        pred_str = mpf.label_to_charac(pred)
        f.write("%s %s %d\n"%(label_str, pred_str, err))
    f.write("error:%4d total:%4d"%(sum(errs), len(labels)))


def get_files_of_dir(dir, file_type = '.mpf', is_full_path = True):
    res = []

    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(file_type):
                continue
            if is_full_path:
                res.append(dir + f)
            else:
                res.append(f)

    return res


def get_test_res_path(dict, test_res_dir = test_res_dir):
    path = test_res_dir
    for key, val in dict.items():
        path += '%s%d_'%(key, val)
    path += 'test.txt'
    return path


def run_online_pca(files, Feature_dim):
    online_pca = OnlinePCA(Feature_dim)

    for f in files:
        print '[OnlinePCA] : ' + f

        mpf.loadFromFile(f)
        if not mpf._is_inited():
            continue
            
        values = numpy.zeros([mpf._sample_num(), mpf._dim()], dtype = floatX)
        mpf.get_values(values, value_normalized)
        if use_bc_metric:
            values = numpy.sqrt(values)

        online_pca.add_batch_data(values)
    
    print '[OnlinePCA] : cal_projection' 
    online_pca.update_projection()

    cPickle.dump(online_pca, open(pca_model_path, 'w'))
    print 'save online_pca to %s'%(pca_model_path)
    return online_pca


def run_online_lda(files, input_dim, online_pca = None):
    online_lda = OnlineLDA(input_dim)

    file_count = 1
    for f in files:
        print '[OnlineLDA] : %s%5d'%(f, file_count)
        file_count += 1

        cluster = numpy.loadtxt(f, dtype = floatX)
        if value_normalized:
            cluster = cluster / numpy.float32(256.)

        if use_bc_metric:
            cluster = numpy.sqrt(cluster)
        
        if online_pca:
            cluster = online_pca.project(cluster, input_dim, whiten = True)

        online_lda.add_cluster(cluster)

    print '[OnlineLDA] : cal_projection' 
    online_lda.update_projection()

    cPickle.dump(online_lda, open(lda_model_path, 'w'))
    print 'save online_lda to %s'%(lda_model_path)

    return online_lda


def run_incrase_mean(files, input_dim, online_pca, PCA_dim, online_lda, LDA_dim, use_bc_metric = True):
    
    increase_mean = IncreaseMean(Label_num, input_dim)

    for f in files:
        print '[IncreaseMean] : ' + f

        mpf.loadFromFile(f)
        if not mpf._is_inited():
            continue
            
        values = numpy.zeros([mpf._sample_num(), mpf._dim()], dtype = floatX)
        mpf.get_values(values, value_normalized)
        if use_bc_metric:
            values = numpy.sqrt(values)

        if online_pca:
            values = online_pca.project(values, PCA_dim, whiten = PCA_whiten)

        if online_lda:
            values = online_lda.project(values, LDA_dim)

        labels = numpy.arange(0, mpf._sample_num(), dtype = 'uint16')
        mpf.get_labels(labels)

        for i in range(0, mpf._sample_num()):
            increase_mean.add_sample(labels[i], values[i, :])

    dict = (increase_mean.get_label2idx_dict(), increase_mean.get_idx2label_dict())
    cPickle.dump(dict, open(dict_model_path, 'w'))
    print 'save dict to %s'%(dict_model_path)

    cPickle.dump(increase_mean, open(knn_model_path, 'w'))
    print 'save knn_model to %s'%(knn_model_path)
    

    return increase_mean


def run_weak_kpc(files, num_candidates, input_dim, online_pca, PCA_dim, online_lda, LDA_dim):
    weak_kpc = WeakKPC(Label_num, input_dim)

    for f in files:
        print '[weak_KPC] : ' + f

        mpf.loadFromFile(f)
        if not mpf._is_inited():
            continue
            
        values = numpy.zeros([mpf._sample_num(), mpf._dim()], dtype = floatX)
        mpf.get_values(values, value_normalized)
        if use_bc_metric:
            values = numpy.sqrt(values)

        if online_pca:
            values = online_pca.project(values, PCA_dim, wirten = PCA_writen)

        if online_lda:
            values = online_lda.project(values, LDA_dim)

        labels = numpy.arange(0, mpf._sample_num(), dtype = 'uint16')
        mpf.get_labels(labels)

        for i in range(0, mpf._sample_num()):
            weak_kpc.add_sample(labels[i], values[i, :])

    weak_kpc.fit(num_candidates)
    
    cPickle.dump(weak_kpc, open(weak_kpc_model_path, 'w'))
    print 'save weak_kpc_model to %s'%(weak_kpc_model_path)


def run_qdf(files, online_pca, PCA_dim, online_lda, LDA_dim, h2):
    qdf = QDF(LDA_dim, Label_num)

    file_count = 1
    for f in files:
        print '[QDF] : %s%5d'%(f, file_count)
        file_count += 1

        cluster = numpy.loadtxt(f, dtype = floatX)
        if value_normalized:
            cluster = cluster / numpy.float32(256.)

        if use_bc_metric:
            cluster = numpy.sqrt(cluster)

        cluster = online_pca.project(cluster, PCA_dim)
        cluster = online_lda.project(cluster, LDA_dim)
        label = mpf.charac_to_label( os.path.basename(f))
        qdf.add_cluster(cluster, label, h2)

    print '[QDF] : cal_projection' 
    qdf.fit()

    cPickle.dump(qdf, open(qdf_model_path, 'w'))
    print 'save mqdf to %s'%(qdf_model_path)


def run_mqdf(files, online_pca, PCA_dim,online_lda, LDA_dim, K, h2):
    mqdf = MQDF(LDA_dim, Label_num, K, h2)

    file_count = 1
    for f in files:
        print '[MQDF] : %s%5d'%(f, file_count)
        file_count += 1

        cluster = numpy.loadtxt(f, dtype = floatX)
        if value_normalized:
            cluster = cluster / numpy.float32(256.)

        if use_bc_metric:
            cluster = numpy.sqrt(cluster)

        cluster = online_pca.project(cluster, PCA_dim)
        cluster = online_lda.project(cluster, LDA_dim)
        label = mpf.charac_to_label( os.path.basename(f))
        mqdf.add_cluster(cluster, label)

    print '[MQDF] : cal_projection' 
    mqdf.fit()

    cPickle.dump(mqdf, open(mqdf_model_path, 'w'))
    print 'save mqdf to %s'%(mqdf_model_path)


def labels_to_index(labels, label2idx_dict):
    indexs = numpy.arange(0, len(labels), dtype = 'int')
    for i in xrange( len(indexs)):
        indexs[i] = label2idx_dict[ labels[i]]
    return indexs


def npc_model_to_nnet_W(increase_mean, label2idx_dict):
    nodes_idxs = labels_to_index(increase_mean.get_labels(), label2idx_dict)
    idxs = numpy.argsort(nodes_idxs)
    nnet_W = increase_mean.get_data()
    return numpy.asarray(nnet_W[idxs, :], dtype = floatX).transpose()


def load_all_data(file_dir = train_dir, batch_size = train_batch_size, trans = None, bias = None, idx_type = True, label2idx_dict = None):

    data_tuple = ()
    index_tuple = ()
    if label2idx_dict is None:
        label2idx_dict, _ = cPickle.load( open(dict_model_path, 'r'))

    files = get_files_of_dir(file_dir)   

    file_count = 1
    for f in files:
        print '[load data] : %s%5d'%(f, file_count)
        file_count += 1

        mpf.loadFromFile(f)
        if not mpf._is_inited():
            continue
            
        values = numpy.zeros([mpf._sample_num(), mpf._dim()], dtype = floatX)
        mpf.get_values(values, value_normalized)
        if use_bc_metric:
            values = numpy.sqrt(values)

        if bias is not None:
            values += bias

        if trans is not None:
            values = numpy.asarray(numpy.dot(values, trans), dtype = floatX)

        data_tuple = data_tuple + (values,)

        labels = numpy.arange(0, mpf._sample_num(), dtype = 'uint16')
        mpf.get_labels(labels)
        if idx_type:
            indexs = labels_to_index(labels, label2idx_dict)
        else:
            indexs = labels      

        index_tuple = index_tuple + (indexs,)


    values = numpy.concatenate(data_tuple)
    print 'data_shape(%d, %d)'%(values.shape[0],values.shape[1])

    targets = numpy.concatenate(index_tuple)
    length = len(targets) -  len(targets) % batch_size

    return values[:length, :], targets[:length]


def shared_dataset(values, labels, borrow = True):
    shared_values = theano.shared(numpy.asarray(values, 
                                                dtype = theano.config.floatX
                                                ),
                                    borrow = borrow
                                    )

    shared_labels = theano.shared(numpy.asarray(labels, 
                                                dtype = theano.config.floatX
                                                ),
                                    borrow = borrow
                                    )

    return shared_values, theano.tensor.cast(shared_labels, 'int32')


def shared_data(values, borrow = True):
    return theano.shared(numpy.asarray(values, 
                                                dtype = theano.config.floatX
                                                ),
                                    borrow = borrow
                                    )

def main(train_func = None, test_func = None):
    if train_func:
        #'''
        print 'start train...'
        start_time = time.time()
        train_func( train_dir)
        end_time = time.time()
        mins = (end_time - start_time) / 60
        secs = (end_time - start_time) % 60
        print 'Train cost:%2dm %2ds\n'%(mins, secs)
        #'''
    
    if test_func:
        print 'start test...'
        start_time = time.time()
        test_func( test_dir, is_log = True)
        end_time = time.time()
        mins = (end_time - start_time) / 60
        secs = (end_time - start_time) % 60
        print 'Test cost:%2dm %2ds'%(mins, secs)
        #'''

    print 'end...'    