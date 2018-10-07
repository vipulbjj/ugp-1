import numpy as np
import numpy.random as rd
import theano
import theano.tensor as tensor


class NetworkLayer(object):
    def __init__(self, input_vector, n_in, n_out, W=None, b=None, activation=tensor.tanh, layer_idx=0):

        assert isinstance(input_vector.__class__, theano.tensor.TensorVariable.__class__)
        self.input_vector = input_vector
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if W is None:
            W = theano.shared(rd.randn(n_in, n_out) / (n_in + n_out), name="W" + str(layer_idx), borrow=True)
        self.W = W

        if b is None:
            b = theano.shared(np.zeros(n_out), name="b" + str(layer_idx), borrow=True)
        self.b = b

        lin_output = tensor.dot(input_vector, self.W) + self.b

        self.output = (lin_output if activation is None
                       else activation(lin_output))

        self.params = [self.W, self.b]

    def compute_output(self, input_values):

        assert input_values.shape[1] == self.n_in

        compute_value = theano.function([self.input_vector], self.output)
        return compute_value(input_values)
