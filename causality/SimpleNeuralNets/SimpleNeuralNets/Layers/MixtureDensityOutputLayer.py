import numpy as np
import numpy.random as rd
import theano
import theano.tensor as tensor


class MixtureDensityOutputLayer(object):
    def __init__(self, input_vector, n_in, dimension_target_variable, n_components, W=None, b=None, layer_idx=0):

        assert isinstance(input_vector.__class__, theano.tensor.TensorVariable.__class__)
        self.input_vector = input_vector
        self.n_in = n_in
        self.n_components = n_components
        self.dimension_target_variable = dimension_target_variable
        self.n_out = (dimension_target_variable + 2) * n_components

        if W is None:
            W = theano.shared(rd.randn(n_in, self.n_out) / (n_in + self.n_out), name="W" + str(layer_idx), borrow=True)
        self.W = W

        if b is None:
            b = theano.shared(np.zeros(self.n_out), name="b" + str(layer_idx), borrow=True)
        self.b = b

        lin_output = tensor.transpose(tensor.dot(input_vector, self.W) + self.b)

        self.mu = lin_output[:n_components*dimension_target_variable]
        self.sigma = tensor.exp(lin_output[n_components*dimension_target_variable : n_components*(dimension_target_variable + 1)])

        mix = tensor.exp(lin_output[n_components*(dimension_target_variable + 1):])
        self.mix = mix / tensor.sum(mix, axis=0)

        self.params = [self.W, self.b]
