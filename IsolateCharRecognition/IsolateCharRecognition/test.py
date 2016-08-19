from Res import *
from Utils import *
import theano
import theano.tensor as T

def test():
    a = theano.shared( numpy.array([[1, 2],[3,4]]))
   
    input_var = T.dvector('input')
    
    #delta = T.reshape(T.repeat(input_var, 2, axis = 0), (1, 2))
    out = T.inc_subtensor(a[0, :], - input_var)
    
    out_fn = theano.function([input_var], outputs = out)
   

    print out_fn(numpy.array([5, 5]))
    
if __name__ == '__main__':
    test()