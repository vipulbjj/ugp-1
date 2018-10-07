import itertools
import numpy as np
import numpy.random as rd
import SimpleNeuralNets.Layers.NetworkLayer as nl
import theano
import theano.tensor as tensor


class FFNN():

    def __init__(self, input_vector, target_vector, n_in, n_hidden, n_out, hid_activations, out_activation, **kwargs):
        if not isinstance(input_vector.__class__, theano.tensor.TensorVariable.__class__):
            raise AssertionError("input_vector needs to be of type 'theano.tensor.TensorVariable'")

        self.normalized_gradient = kwargs.get('normalized_gradient', False)

        self.input_vector = input_vector
        self.target_vector = target_vector
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.layers = self._wire_layers(hid_activations, input_vector, n_hidden, n_in, n_out, out_activation)
        self.output = self.layers.get("layer_"+str(len(n_hidden))).output
        self.params = list(itertools.chain(*[layer.params for layer in self.layers.itervalues()]))

    def _wire_layers(self, hid_activations, input_vector, n_hidden, n_in, n_out, out_activation):
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
        layers["layer_" + str(len(n_hidden))] = nl.NetworkLayer(layers.get("layer_" + str(len(n_hidden) - 1)).output,
                                                                n_hidden[-1],
                                                                n_out,
                                                                activation=out_activation,
                                                                layer_idx=len(n_hidden))
        return layers

    def compute_layer(self, input_values, layer_idx):
        compute = theano.function([self.input_vector], self.layers.get("layer_" + str(layer_idx)).output)
        return compute(input_values)

    def predict(self, input_values):
        compute = theano.function([self.input_vector], self.output)
        return compute(input_values)

    def print_network_graph(self):
        theano.printing.pydotprint(self.output,
                                   var_with_name_simple=True,
                                   compact=True,
                                   outfile='nn-theano-forward_prop.png',
                                   format='png')

    def loss_function(self, n_samples, l1_regularization_strength, l2_regularization_strength):
        # if not isinstance(regularization_strength, float): # theano fails silently if it is integer :(
        #     raise AssertionError('regluarization_strength needs to be float.')

        loss = tensor.mean(tensor.sqr(self.output.reshape([n_samples, ]) - self.target_vector))

        l1_reg_loss = tensor.sum(np.abs(self.layers.values()[0].W))
        for layer in self.layers.values()[1:]:
            l1_reg_loss += tensor.sum(np.abs(layer.W))

        l2_reg_loss = tensor.sum(tensor.sqr(self.layers.values()[0].W))
        for layer in self.layers.values()[1:]:
            l2_reg_loss += tensor.sum(tensor.sqr(layer.W))

        l1_regularization = 1/n_samples * l1_regularization_strength/2 * l1_reg_loss

        l2_regularization = 1/n_samples * l2_regularization_strength/2 * l2_reg_loss

        return loss + l1_regularization + l2_regularization

    def compute_param_updates(self, n_samples, l1_regularization_strength, l2_regularization_strength, lr):
        gparams = []
        for param in self.params:
            gparam = tensor.grad(self.loss_function(n_samples, l1_regularization_strength, l2_regularization_strength), param)

            if self.normalized_gradient:
                gparam = gparam / tensor.sqrt(tensor.sum(tensor.sqr(gparam)))

            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * lr))

        return updates

    def get_gradient(self, n_samples, l1_regularization_strength, l2_regularization_strength, lr):
        return theano.function([self.input_vector, self.target_vector],
                               self.loss_function(n_samples, l1_regularization_strength, l2_regularization_strength),
                               updates=tuple(self.compute_param_updates(n_samples, l1_regularization_strength, l2_regularization_strength, lr)))

    def train(self, input_values, target_values, test_input=None, test_target=None, **kwargs):

        n_iterations = kwargs.get('n_iterations', 10000)
        print_loss = kwargs.get('print_loss', False)
        learning_rate = kwargs.get('learning_rate', 0.01)
        l1_regularization_strength = kwargs.get('l1_strength', 0.0)
        l2_regularization_strength = kwargs.get('l2_strength', 0.0)
        sigma_weight_init = kwargs.get('sigma_weight_init', 1.0)
        bias_weight_init = kwargs.get('sigma_bias_init', 0.0)

        gradient_step = self.get_gradient(input_values.shape[0],  l1_regularization_strength, l2_regularization_strength, learning_rate)
        calculate_loss = theano.function([self.input_vector, self.target_vector], self.loss_function(input_values.shape[0],  l1_regularization_strength, l2_regularization_strength))

        if test_input is not None:
            calculate_test_loss = theano.function([self.input_vector, self.target_vector], self.loss_function(test_input.shape[0],  l1_regularization_strength, l2_regularization_strength))

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
                test_losses.append(calculate_test_loss(test_input, test_target))

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" %(i, calculate_loss(input_values, target_values))
                print "gradient after iteration " + str(i) + ": ", learning_rate * grad_step
                print "\n"

        return losses, test_losses
