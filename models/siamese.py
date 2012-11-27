import numpy
import theano
from theano import tensor
from noisy_encoder.utils.corruptions import BinomialCorruptorScaledGroup, BinomialCorruptorScaled
from noisy_encoder.models.mlp_new import DropOutMLP
from pylearn2.corruption import GaussianCorruptor
from pylearn2.utils import sharedX, serial

class Siamese(object):

    def __init__(self, numpy_rng,
                    theano_rng,
                    image_topo,
                    base_model,
                    n_units,
                    input_corruption_levels,
                    hidden_corruption_levels,
                    n_outs,
                    act_enc,
                    irange,
                    bias_init,
                    method = 'diff',
                    fine_tune = False,
                    rng = 9001):


        self.theano_rng = theano_rng
        self.numpy_rng = numpy_rng

        self.x = tensor.matrix('x')
        self.x_p = tensor.matrix('x_p')
        self.y = tensor.ivector('y')


        # make corruptors:
        input_corruptors = []
        for item in input_corruption_levels:
            if item == None or item == 0.0:
                input_corruptors.extend([None])
            else:
                input_corruptors.extend([GaussianCorruptor(corruption_level = item)])

        hidden_corruptors = []
        for item in hidden_corruption_levels:
            if item == None or item == 0.0:
                hidden_corruptors.extend([None])
            else:
                hidden_corruptors.extend([BinomialCorruptorScaled(corruption_level = item)])



        # load base model
        base_model = serial.load(base_model)

        inputs = self.x.reshape(image_topo)
        inputs_p = self.x_p.reshape(image_topo)
        inputs = base_model(inputs)
        inputs_p = base_model(inputs_p)

        if method == 'diff':
            self.inputs = inputs - inputs_p
        elif method == 'kl':
            self.inputs = inputs * tensor.log(inputs_p) + \
                            (1 - inputs) * tensor.log(1 - inputs_p)
        else:
            raise NameError("Unknown method: {}".format(method))

        self.mlp = DropOutMLP(input_corruptors = input_corruptors,
                            hidden_corruptors = hidden_corruptors,
                            n_units = n_units,
                            n_outs = n_outs,
                            act_enc = act_enc,
                            irange = irange,
                            bias_init = bias_init)

        if fine_tune:
            self.params = base_model.model.mlp.hiddens._params + self.mlp._params
        else:
            self.params = self.mlp._params

    def negative_log_likelihood(self, x, y):
        return -tensor.mean(tensor.log(self.mlp.p_y_given_x(x))[tensor.arange(y.shape[0]), y])

    def errors(self, x, y):
        return tensor.mean(tensor.neq(self.mlp.predict_y(x), y))


    def build_finetune_functions(self, datasets, batch_size, w_l1_ratio, act_l1_ratio, enable_momentum):

        (train_set_x, train_set_y) = datasets[0]
        (train_set_x_p, train_set_y_p) = datasets[1]
        (valid_set_x, valid_set_y) = datasets[2]
        (valid_set_x_p, valid_set_y_p) = datasets[3]
        (test_set_x, test_set_y) = datasets[4]
        (test_set_x_p, test_set_y_p) = datasets[5]


        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = tensor.lscalar('index')  # index to a [mini]batch
        learning_rate = tensor.scalar('lr')
        if enable_momentum is None:
            momentum = None
        else:
            momentum = tensor.scalar('momentum')

        # compute the gradients with respect to the model parameters
        w_l1 = tensor.abs_(self.mlp.hiddens.layers[-1].weights).mean() * w_l1_ratio
        cost = self.negative_log_likelihood(self.inputs, self.y) + w_l1
        gparams = tensor.grad(cost, self.params)
        errors = self.errors(self.inputs, self.y)

        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self.params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.params, gparams):
                inc = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc - learning_rate * gparam
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[cost, errors],
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.x_p: train_set_x_p[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})


        test_score_i = theano.function([index], errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.x_p: test_set_x_p[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.x_p: valid_set_x_p[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score
