import numpy
import lasagne
import theano
import theano.tensor as T
from Res import *


def build_mlp(input_var = None, W_pca = None, W_lda = None, W_npc = None):
    network = lasagne.layers.InputLayer(shape = (None, Feature_dim),
                                        input_var = input_var)

    if W_pca is None:
        network = lasagne.layers.DenseLayer(network,
                                            num_units = PCA_dim,
                                            b = None,
                                            nonlinearity = lasagne.nonlinearities.sigmoid
                                            )
    else:
        network = lasagne.layers.DenseLayer(network,
                                            num_units = PCA_dim,
                                            W = W_pca,
                                            b = None,
                                            nonlinearity = lasagne.nonlinearities.linear
                                            )

    #network = lasagne.layers.DropoutLayer(network, p = 0.1)

    if W_lda is None:
        network = lasagne.layers.DenseLayer(network,
                                            num_units = LDA_dim,
                                            b = None,
                                            nonlinearity = lasagne.nonlinearities.linear
                                            )
    else:
        network = lasagne.layers.DenseLayer(network,
                                            num_units = LDA_dim,
                                            W = W_lda,
                                            b = None,
                                            nonlinearity = lasagne.nonlinearities.linear
                                            )
    #network = lasagne.layers.DropoutLayer(network, p = 0.2)

    if W_npc is None:
        network = lasagne.layers.DenseLayer(network,
                                            num_units = Label_num,
                                            nonlinearity = lasagne.nonlinearities.softmax,
                                            b = None
                                            )
    else:
        network = lasagne.layers.DenseLayer(network,
                                            num_units = Label_num,
                                            nonlinearity = lasagne.nonlinearities.softmax,
                                            W = W_npc,
                                            )

    return network
