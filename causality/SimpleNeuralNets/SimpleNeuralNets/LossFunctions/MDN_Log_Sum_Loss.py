import numpy as np
import theano
import theano.tensor as tensor


class Log_Sum_Loss(object):

    def __init__(self, input_vector, target_vector):
        self.input_vector = input_vector
        self.target_vector = target_vector
        # self.mix = mix
        # self.mu = mean
        # self.sigma = sigma

    def loss(self, n_samples, regularization_strength, mix, mu, sigma):
        log_sum_loss = -tensor.sum(tensor.log(
                            tensor.sum(mix * tensor.inv(np.sqrt(2 * np.pi) * sigma) *
                                       tensor.exp(tensor.neg(tensor.sqr(mu - self.target_vector)) *
                                                  tensor.inv(2 * tensor.sqr(sigma))), axis=0)
        ))

        # reg_loss = tensor.sum(tensor.sqr(self.layers.values()[0].W))
        # for layer in self.layers.values()[1:]:
        #     reg_loss += tensor.sum(tensor.sqr(layer.W))

        # regularization = 1/n_samples * regularization_strength/2 * reg_loss

        return log_sum_loss #+ regularization

    def _compute_param_updates(self, n_samples, regularization_strength, lr, params, mix, mu, sigma):
        gparams = []
        for param in params:
            gparam = tensor.grad(self.loss(n_samples, regularization_strength, mix, mu, sigma), param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(params, gparams):
            updates.append((param, param - gparam * lr))

        return updates

    def get_gradient(self, n_samples, regularization_strength, lr, params, mix, mu, sigma):
        return theano.function([self.input_vector, self.target_vector],
                               self.loss(n_samples, regularization_strength, mix, mu, sigma),
                               updates=tuple(self._compute_param_updates(n_samples, regularization_strength, lr, params, mix, mu, sigma)))
