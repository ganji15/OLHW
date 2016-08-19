import os, time, cPickle
import lasagne
import theano
import theano.tensor as T
from Res import *
from Utils import main, get_files_of_dir, get_test_res_path, load_all_data, npc_model_to_nnet_W, shared_data, shared_dataset
from BuildNet import *


def train(train_dir):
    index = T.lscalar('index')
    input_var = T.fmatrix('inputs')
    target_var = T.ivector('targets')

    online_pca = cPickle.load(open(pca_model_path, 'r'))
    online_lda = cPickle.load(open(lda_model_path, 'r'))
    increase_mean = cPickle.load(open(knn_model_path, 'r'))
    (label2idx_dict, idx2label_dict) = cPickle.load(open(dict_model_path, 'r'))
    W_pca = shared_data( online_pca.get_project_cov(PCA_dim))
    W_lda = shared_data( online_lda.get_project_cov(LDA_dim))
    W_npc = shared_data( npc_model_to_nnet_W(increase_mean, label2idx_dict))
    data_bias = -online_pca.get_mean()

    batch_size = train_batch_size
    x_train, y_train = load_all_data(train_dir, batch_size, bias = data_bias)
    num_train_batches =  x_train.shape[0] / batch_size
    shared_x_train, shared_y_train = shared_dataset(x_train, y_train)

    network = build_mlp(input_var, W_pca, W_lda, W_npc)
    prediction = lasagne.layers.get_output(network)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable = True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([index],
                                outputs = [loss, acc],
                                updates = updates,
                                givens={
                                        input_var: shared_x_train[index * batch_size: (index + 1) * batch_size],
                                        target_var: shared_y_train[index * batch_size: (index + 1) * batch_size]
                                        }
                                )
    
    best_acc = 0.0
    for epoch in xrange(num_epoch):
        train_acc = 0
        train_loss = 0
        start_time = time.time()

        random_batches = range(0, num_train_batches)
        numpy.random.shuffle( random_batches)

        for mini_batch in random_batches:
            ls, acc= train_fn(mini_batch)
            train_loss += ls
            train_acc += acc

        train_acc = train_acc / num_train_batches
        
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epoch, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_loss / num_train_batches))
        print("  training accuracy:\t\t{:.2f} %\n".format(train_acc * 100))

        if epoch % 10 == 0 and train_acc > best_acc:
            cPickle.dump((input_var, network, data_bias), open(mlp_model_path, 'w'))
            print 'Epoch: %d save network to path %s'%(epoch, mlp_model_path)

        
    print 'train over' 


def test(test_dir, is_log = True):
    
    index = T.lscalar('index2')
    target_var = T.ivector('targets2')
    
    (input_var, network, data_bias) = cPickle.load(open(mlp_model_path, 'r'))
    prediction = lasagne.layers.get_output(network)
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    batch_size = test_batch_size
    x_test, y_test = load_all_data(test_dir, batch_size, data_bias)
    num_test_batches =  x_test.shape[0] / batch_size
    shared_x_test, shared_y_test = shared_dataset(x_test, y_test)

    test_fn = theano.function([index],
                                outputs = acc,
                                givens={
                                        input_var: shared_x_test[index * batch_size: (index + 1) * batch_size],
                                        target_var: shared_y_test[index * batch_size: (index + 1) * batch_size]
                                        }
                                )

    acc = 0.0
    for i in xrange(num_test_batches):
        acc += test_fn(i)

    print 'test acc %.2f%%'%(acc / num_test_batches  * 100)

if __name__ == '__main__':
    main(train, test)