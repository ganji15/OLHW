import cPickle
from Res import *
from Utils import *
import numpy
import theano
import theano.tensor as T

def update_onestep2(shared_means, sample, target, p, sigma):
    dists = T.sum( T.square(shared_means - sample), axis = 1)
    mr_idx = T.argmin(dists)
    
    if mr_idx == target:
        return shared_means, T.as_tensor(0.0)

    mk = shared_means[target, :]  
    mr = shared_means[mr_idx, :]
    
    exp_delta_k = T.exp(- sigma * (T.sum( T.square(sample - mk)) - T.sum( T.square(sample - mr))))
    mk_delta = - 2 * sigma * exp_delta_k * (sample - mk) / T.square(1 + exp_delta_k)
    mr_delta = 2 * sigma * exp_delta_k * (sample - mr) / T.square(1 + exp_delta_k)

    '''
    shared_means = T.inc_subtensor(shared_means[target], - p * mk_delta)
    shared_means = T.inc_subtensor(shared_means[mr], - p * mr_delta)
    '''

    shared_means = T.set_subtensor(shared_means[target, :], shared_means[target, :] - p * mk_delta)
    shared_means = T.set_subtensor(shared_means[mr_idx, :], shared_means[mr_idx, :] - p * mr_delta)

    loss = 1 / (1 + exp_delta_k)

    return T.cast(shared_means, dtype = theano.config.floatX), loss



def update_onestep(shared_means, sample, target, p, sigma):
    dists = T.sum( T.square(shared_means - sample), axis = 1)
    idxs = T.argsort(dists)[:2]
    
    mk = shared_means[target, :]
    mr_idx = idxs[1] * T.eq(idxs[0], target) + idxs[0] * T.neq(idxs[0], target)
    mr = shared_means[mr_idx, :]

    exp_delta_k = T.exp(- sigma * (T.sum( T.square(sample - mk)) - T.sum( T.square(sample - mr))))
    mk_delta = - 2 * sigma * exp_delta_k * (sample - mk) / T.square(1 + exp_delta_k)
    mr_delta = 2 * sigma * exp_delta_k * (sample - mr) / T.square(1 + exp_delta_k)
    

    shared_means = T.set_subtensor(shared_means[target, :], shared_means[target, :] - p * mk_delta)
    shared_means = T.set_subtensor(shared_means[mr_idx, :], shared_means[mr_idx, :] - p * mr_delta)
    

    loss = 1 / (1 + exp_delta_k)

    return T.cast(shared_means, dtype = theano.config.floatX), loss


def run_lvq(train_dir = train_dir):
    PCA_dim = 300
    LDA_dim = 150
    online_pca = cPickle.load(open(pca_model_path, 'r'))
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    increase_mean = cPickle.load( open(knn_model_path, 'r'))
    data_bias = -online_pca.get_mean()

    trans = numpy.dot(online_pca.get_project_cov(PCA_dim), online_lda.get_project_cov(LDA_dim))
    values, idxs = load_all_data(train_dir, trans = trans, label2idx_dict = increase_mean.get_label2idx_dict(), bias = data_bias)
    
    shared_x, shared_idxs = shared_dataset(values, idxs)
    shared_means = shared_data( increase_mean.get_data())
    
    input_var = T.fvector('input')
    target_var = T.iscalar('target')
    index = T.lscalar('index')

    p = theano.shared(0.1)
    sigma = theano.shared(0.1)
    change_rate = 1e-4
    new_means, loss = update_onestep2(shared_means, input_var, target_var, p, sigma)
    updates = [(shared_means, new_means)]

    update_means = theano.function([index], 
                                    outputs = loss,
                                    updates = updates,
                                    givens={
                                            input_var: shared_x[index],
                                            target_var: shared_idxs[index]
                                            }
                                 )

    update_vargins = theano.function([], outputs = [], updates = [(p, p * (1 - change_rate)), (sigma, sigma* (1 + change_rate))])

    num_epoch = 100
    for epoch in xrange(num_epoch):
        indices = range(0, len(idxs))
        numpy.random.shuffle(indices)

        start_time = time.time()
        loss = 0
        for i in indices:
            loss += update_means(i)

        loss = loss / len(idxs)
        update_vargins()
        print 'epoch %d: cost %.3fs, loss %.6f'%(epoch, time.time() - start_time, loss)
        cPickle.dump((shared_means.eval(), increase_mean.get_labels()), open(lvq_model_path, 'w'))

    cPickle.dump(lvq, open('D:\\lvq.pkl', w))

if __name__ == '__main__':
    run_lvq()