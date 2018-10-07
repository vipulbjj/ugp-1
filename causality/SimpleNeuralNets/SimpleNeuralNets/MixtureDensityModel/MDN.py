import itertools
import numpy as np
import numpy.random as rd
import SimpleNeuralNets.Layers.NetworkLayer as nl
import SimpleNeuralNets.Layers.MixtureDensityOutputLayer as mdl
import theano
import theano.tensor as tensor


class MDN(object):

    def __init__(self,
                 input_vector,
                 target_vector,
                 n_in,
                 n_hidden,
                 dimension_target_variable,
                 number_of_components,
                 hid_activations,
                 **kwargs):
        if not isinstance(input_vector.__class__, theano.tensor.TensorVariable.__class__):
            raise AssertionError("input_vector needs to be of type 'theano.tensor.TensorVariable'")

        self.normalized_gradient = kwargs.get('normalized_gradient', False)

        self.input_vector = input_vector
        self.target_vector = target_vector
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.number_of_components = number_of_components
        self.dimension_target_variable = dimension_target_variable
        self.n_out = (dimension_target_variable + 2) * number_of_components
        self.layers = self._wire_layers(hid_activations, input_vector, n_hidden, n_in)
        self.mu = self.layers.get("layer_"+str(len(n_hidden))).mu
        self.sigma = self.layers.get("layer_"+str(len(n_hidden))).sigma
        self.mix = self.layers.get("layer_"+str(len(n_hidden))).mix
        self.params = list(itertools.chain(*[layer.params for layer in self.layers.itervalues()]))

    def _wire_layers(self, hid_activations, input_vector, n_hidden, n_in):
        layers = dict()
        layers["layer_0"] = nl.NetworkLayer(input_vector,
                                            n_in,
                                            n_hidden[0],
                                            activation=hid_activations[0],
                                            layer_idx=0)
        for idx in range(1, len(n_hidden)):
            layers["layer_" + str(idx)] = nl.NetworkLayer(layers.get("layer_" + str(idx - 1)).output,
                                                          n_hidden[idx - 1],
                                                          n_hidden[idx],
                                                          activation=hid_activations[idx],
                                                          layer_idx=idx)
        layers["layer_" + str(len(n_hidden))] = \
            mdl.MixtureDensityOutputLayer(layers.get("layer_" + str(len(n_hidden) - 1)).output,
                                                     n_hidden[-1],
                                                     dimension_target_variable=self.dimension_target_variable,
                                                     n_components=self.number_of_components,
                                                     layer_idx=len(n_hidden))
        return layers

    def compute_layer(self, input_values, layer_idx):
        compute = theano.function([self.input_vector], self.layers.get("layer_" + str(layer_idx)).output)
        return compute(input_values)

    def predict_mu(self, input_values):
        compute_mu = theano.function([self.input_vector], self.mu)
        return compute_mu(input_values)

    def predict_sigma(self, input_values):
        compute_sigma = theano.function([self.input_vector], self.sigma)
        return compute_sigma(input_values)

    def predict_mix(self, input_values):
        compute_mix = theano.function([self.input_vector], self.mix)
        return compute_mix(input_values)

    def predict_params(self, input_values):
        return self.predict_mu(input_values), self.predict_sigma(input_values), self.predict_mix(input_values)

    def print_network_graph(self):
        theano.printing.pydotprint(self.predict_params,
                                   var_with_name_simple=True,
                                   compact=True,
                                   outfile='nn-theano-forward_prop.png',
                                   format='png')

    def logsum_loss(self, n_samples, l1_regularization_strength, l2_regularization_strength):
        log_sum_loss = -tensor.sum(tensor.log(
                            tensor.sum(self.mix * tensor.inv(np.sqrt(2 * np.pi) * self.sigma) *
                                       tensor.exp(tensor.neg(tensor.sqr(self.mu - self.target_vector)) *
                                                  tensor.inv(2 * tensor.sqr(self.sigma))), axis=0)
        ))

        l1_reg_loss = tensor.sum(np.abs(self.layers.values()[0].W))
        for layer in self.layers.values()[1:]:
            l1_reg_loss += tensor.sum(np.abs(layer.W))

        l2_reg_loss = tensor.sum(tensor.sqr(self.layers.values()[0].W))
        for layer in self.layers.values()[1:]:
            l2_reg_loss += tensor.sum(tensor.sqr(layer.W))

        l1_regularization = 1/n_samples * l1_regularization_strength/2 * l1_reg_loss

        l2_regularization = 1/n_samples * l2_regularization_strength/2 * l2_reg_loss

        return log_sum_loss + l1_regularization + l2_regularization

    def compute_param_updates(self, n_samples, l1_regularization_strength, l2_regularization_strength, lr):
        gparams = []
        for param in self.params:
            gparam = tensor.grad(self.logsum_loss(n_samples, l1_regularization_strength, l2_regularization_strength), param)

            if self.normalized_gradient:
                gparam = gparam / tensor.sqrt(tensor.sum(tensor.sqr(gparam)))

            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * lr))

        return updates

    def get_gradient(self, n_samples, l1_regularization_strength, l2_regularization_strength, lr):
        return theano.function([self.input_vector, self.target_vector],
                               self.logsum_loss(n_samples, l1_regularization_strength, l2_regularization_strength),
                               updates=tuple(self.compute_param_updates(n_samples, l1_regularization_strength, l2_regularization_strength, lr)))

    def train(self, input_values, target_values, test_input=None, test_target=None, **kwargs):

        n_iterations = kwargs.get('n_iterations', 10000)
        print_loss = kwargs.get('print_loss', False)
        learning_rate = kwargs.get('learning_rate', 0.01)
        l1_regularization_strength = kwargs.get('l1_strength', 0.0)
        l2_regularization_strength = kwargs.get('l2_strength', 0.0)
        sigma_weight_init = kwargs.get('sigma_weight_init', 1.0)
        bias_weight_init = kwargs.get('sigma_bias_init', 0.0)

        gradient_step = self.get_gradient(input_values.shape[0], l1_regularization_strength, l2_regularization_strength, learning_rate)

        calculate_loss = theano.function([self.input_vector, self.target_vector], self.logsum_loss(input_values.shape[0], l1_regularization_strength, l2_regularization_strength))
        calculate_sigma = theano.function([self.input_vector], self.sigma)
        calculate_mu = theano.function([self.input_vector], self.mu)
        calculate_mix = theano.function([self.input_vector], self.mix)

        losses = []
        test_losses = []

        # reinitialize weights
        for layer in self.layers.values():
            layer.W.set_value(sigma_weight_init * rd.randn(layer.n_in, layer.n_out) + bias_weight_init)
            layer.b.set_value(np.zeros(layer.n_out))

        for i in xrange(0, n_iterations):
            # This will update our parameters W2, b2, W1 and b1!
            grad_step = gradient_step(input_values, target_values)
            losses.append(calculate_loss(input_values, target_values))
            if test_input is not None:
                test_losses.append(calculate_loss(test_input, test_target))

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" % (i, calculate_loss(input_values, target_values))
                print "Sigma after iteration " + str(i) + ": ", calculate_sigma(input_values).min(axis=1), calculate_sigma(input_values).max(axis=1)
                print "Mu after iteration " + str(i) + ": ", calculate_mu(input_values).min(axis=1), calculate_mu(input_values).max(axis=1)
                print "mix after iteration " + str(i) + ": ", calculate_mix(input_values).min(axis=1), calculate_mix(input_values).max(axis=1)
                print "gradient after iteration " + str(i) + ": ", learning_rate * grad_step
                print "\n\n"

        return losses, test_losses

    def _gaussian(self, x, mu, sigma):
        return np.exp(-np.power(x - mu, 2)/(2 * np.power(sigma, 2))) / np.sqrt(2 * np.pi * np.power(sigma, 2))

    def gaussian_array(self, x, y, mu, sigma, mix):
        n_dim = mu.shape[0]
        lst = []
        for idx in range(len(x)):
            val = 0
            for dim in range(n_dim):
                val += mix[dim, idx] * self._gaussian(y, mu[dim, idx], sigma[dim, idx])
            lst.append(val)
        return np.meshgrid(x, y), np.array(lst).T
