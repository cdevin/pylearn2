import numpy
import theano
from theano import tensor
from noisy_encoder.utils.corruptions import BinomialCorruptorScaledGroup, BinomialCorruptorScaled
from noisy_encoder.models.mlp import DropOutMLP
from noisy_encoder.models.dropouts import DeepDropOutHiddenLayer
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


    def __call__(self):
        return self.mlp.p_y_given_x(self.inputs)


    def build_finetune_functions(self, datasets, batch_size, coeffs, enable_momentum):

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
        w_l1 = tensor.abs_(self.mlp.hiddens.layers[-1].weights).mean() * coeffs['w_l1']
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

class SiameseMix(object):

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
        self.y_ = tensor.matrix('y_')
        self.inputs = self.x.reshape(image_topo)
        self.inputs_p = self.x_p.reshape(image_topo)


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
        self.conv = base_model.conv
        self.conv_hiddens = base_model.hiddens
        self.conv_loglayer = base_model.loglayer


        self.mlp = DropOutMLP(input_corruptors = input_corruptors,
                            hidden_corruptors = hidden_corruptors,
                            n_units = n_units,
                            n_outs = n_outs,
                            act_enc = act_enc,
                            irange = irange,
                            bias_init = bias_init)


        self.conv_params = self.conv._params + self.conv_hiddens._params + self.conv_loglayer._params
        self.siamese_params = self.conv._params + self.conv_hiddens._params + self.mlp._params

        self.conv_wl1 = tensor.sum([abs(item.weights).sum() for item in \
                self.conv_hiddens.layers]) + abs(self.conv_loglayer.layers[0].weights).sum()
        self.conv_wl2 = tensor.sum([(item.weights ** 2).sum() for item in \
                self.conv_hiddens.layers]) + (self.conv_loglayer.layers[0].weights ** 2).sum()

    def conv_encode(self, inputs):
        return self.conv(inputs).flatten(2)

    def conv_hiddens_encode(self, inputs):
        return self.conv_hiddens.encode(self.conv_encode(inputs))

    def test_conv_hiddens_encode(self, inputs):
        return self.conv_hiddens.test_encode(self.conv_encode(inputs))

    def conv_cross_entropy(self, x, y):
        h = self.conv_loglayer.encode(self.conv_hiddens_encode(x))
        h_ = tensor.switch(tensor.lt(h, 0.00000001), -10, tensor.log(h))
        h_1 = tensor.switch(tensor.lt(1-h, 0.00000001), -10, tensor.log(1-h))
        return -(y * h_ + (1-y)*h_1).sum(axis=1).mean()

    def conv_errors(self, x, y):
        y_pred = tensor.argmax(self.conv_loglayer.test_encode(self.test_conv_hiddens_encode(x)), axis =1)
        return tensor.mean(tensor.neq(y[tensor.arange(y.shape[0]), y_pred], tensor.ones_like(y_pred)))

    def siamese_encode(self):
        h1 = self.test_conv_hiddens_encode(self.inputs)
        h2 = self.test_conv_hiddens_encode(self.inputs_p)
        return h1 - h2

    def siamese_negative_log_likelihood(self, y):
        h = self.siamese_encode()
        return -tensor.mean(tensor.log(self.mlp.p_y_given_x(h))[tensor.arange(y.shape[0]), y])

    def siamese_errors(self, y):
        h = self.siamese_encode()
        return tensor.mean(tensor.neq(self.mlp.predict_y(h), y))

    def __call__(self):
        h = self.siamese_encode()
        return self.mlp.p_y_given_x(h)


    def build_finetune_functions(self, datasets, batch_size, coeffs, enable_momentum):
        """
        In this model with have two set of mini-batches for training. One set updates
        the simase path, one just the old conv path.
        """

        siamese_datasets, conv_datasets = datasets
        (conv_train_set_x, conv_train_set_y) = conv_datasets[0]
        (conv_valid_set_x, conv_valid_set_y) = conv_datasets[1]
        (conv_test_set_x, conv_test_set_y) = conv_datasets[2]

        (siamese_train_set_x, siamese_train_set_y) = siamese_datasets[0]
        (siamese_train_set_x_p, siamese_train_set_y_p) = siamese_datasets[1]


        index = tensor.lscalar('index')  # index to a [mini]batch
        learning_rate = tensor.scalar('lr')
        if enable_momentum is None:
            momentum = None
        else:
            momentum = tensor.scalar('momentum')

        ## siamese
        cost = self.siamese_negative_log_likelihood(self.y)
        gparams = tensor.grad(cost, self.siamese_params)
        errors = self.siamese_errors(self.y)

        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self.siamese_params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.siamese_params, gparams):
                inc_s = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc_s - learning_rate * gparam
                updates[inc_s] = updated_inc
                updates[param] = param + updated_inc


        ## conv
        conv_cost = self.conv_cross_entropy(self.inputs, self.y_)
        conv_cost += coeffs['conv_w_l1'] * self.conv_wl1 + coeffs['conv_w_l2'] * self.conv_wl2
        conv_gparams = tensor.grad(conv_cost, self.conv_params)
        conv_errors = self.conv_errors(self.inputs, self.y_)

        # compute list of fine-tuning updates
        conv_updates = {}
        if momentum is None:
            for param, gparam in zip(self.conv_params, conv_gparams):
                conv_updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.conv_params, conv_gparams):
                inc_c = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc_c - learning_rate * gparam
                conv_updates[inc_c] = updated_inc
                conv_updates[param] = param + updated_inc


        # theano functions
        train_fn_siamese = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[cost, errors],
              updates=updates,
              givens={
                self.x: siamese_train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.x_p: siamese_train_set_x_p[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: siamese_train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        train_fn_conv = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[conv_cost, conv_errors],
              updates=conv_updates,
              givens={
                self.x: conv_train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y_: conv_train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})


        test_score_i = theano.function([index], conv_errors,
                 givens={
                   self.x: conv_test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y_: conv_test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], conv_errors,
              givens={
                 self.x: conv_valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y_: conv_valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # for now valid and test score and conv path on lisa dataset
        def valid_score():
            return [valid_score_i(i) for i in xrange(conv_valid_set_x.get_value(borrow=True).shape[0] / batch_size)]

        def test_score():
            return [test_score_i(i) for i in xrange(conv_test_set_x.get_value(borrow = True).shape[0] / batch_size)]

        return train_fn_conv, train_fn_siamese, valid_score, test_score

class SiameseMixSingleCategory(object):

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
        self.y_ = tensor.ivector('y_')
        self.inputs = self.x.reshape(image_topo)
        self.inputs_p = self.x_p.reshape(image_topo)


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
        self.conv = base_model.conv
        self.conv_mlp = base_model.mlp

        self.mlp = DropOutMLP(input_corruptors = input_corruptors,
                            hidden_corruptors = hidden_corruptors,
                            n_units = n_units,
                            n_outs = n_outs,
                            act_enc = act_enc,
                            irange = irange,
                            bias_init = bias_init)


        self.conv_params = self.conv._params + self.conv_mlp._params
        self.siamese_params = self.conv._params + self.conv_mlp.hiddens._params + self.mlp._params

        self.conv_wl1 = tensor.sum([abs(item).sum() for item in \
                self.conv_mlp.weights])
        self.conv_wl2 = tensor.sum([(item ** 2).sum() for item in \
                self.conv_mlp.weights])

    def conv_encode(self, inputs):
        return self.conv(inputs).flatten(2)

    def conv_hiddens_encode(self, inputs):
        return self.conv_mlp.hiddens.encode(self.conv_encode(inputs))

    def test_conv_hiddens_encode(self, inputs):
        return self.conv_mlp.hiddens.test_encode(self.conv_encode(inputs))

    def conv_cross_entropy(self, x, y):
        h = self.conv_mlp.p_y_given_x(self.conv_encode(x))
        return -tensor.mean(tensor.log(h)[tensor.arange(y.shape[0]), y])

    def conv_errors(self, x, y):
        return tensor.mean(tensor.neq(self.conv_mlp.predict_y(self.conv_encode(x)), y))

    def siamese_encode(self):
        h1 = self.test_conv_hiddens_encode(self.inputs)
        h2 = self.test_conv_hiddens_encode(self.inputs_p)
        return h1 - h2

    def siamese_negative_log_likelihood(self, y):
        h = self.siamese_encode()
        return -tensor.mean(tensor.log(self.mlp.p_y_given_x(h))[tensor.arange(y.shape[0]), y])

    def siamese_errors(self, y):
        h = self.siamese_encode()
        return tensor.mean(tensor.neq(self.mlp.predict_y(h), y))

    def __call__(self):
        h = self.siamese_encode()
        return self.mlp.p_y_given_x(h)


    def build_finetune_functions(self, datasets, batch_size, coeffs, enable_momentum):
        """
        In this model with have two set of mini-batches for training. One set updates
        the simase path, one just the old conv path.
        """

        siamese_datasets, conv_datasets = datasets
        (conv_train_set_x, conv_train_set_y) = conv_datasets[0]
        (conv_valid_set_x, conv_valid_set_y) = conv_datasets[1]
        (conv_test_set_x, conv_test_set_y) = conv_datasets[2]

        (siamese_train_set_x, siamese_train_set_y) = siamese_datasets[0]
        (siamese_train_set_x_p, siamese_train_set_y_p) = siamese_datasets[1]


        index = tensor.lscalar('index')  # index to a [mini]batch
        learning_rate = tensor.scalar('lr')
        if enable_momentum is None:
            momentum = None
        else:
            momentum = tensor.scalar('momentum')

        ## siamese
        cost = self.siamese_negative_log_likelihood(self.y)
        gparams = tensor.grad(cost, self.siamese_params)
        errors = self.siamese_errors(self.y)

        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self.siamese_params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.siamese_params, gparams):
                inc_s = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc_s - learning_rate * gparam
                updates[inc_s] = updated_inc
                updates[param] = param + updated_inc


        ## conv
        conv_cost = self.conv_cross_entropy(self.inputs, self.y_)
        conv_cost += coeffs['conv_w_l1'] * self.conv_wl1 + coeffs['conv_w_l2'] * self.conv_wl2
        conv_gparams = tensor.grad(conv_cost, self.conv_params)
        conv_errors = self.conv_errors(self.inputs, self.y_)

        # compute list of fine-tuning updates
        conv_updates = {}
        if momentum is None:
            for param, gparam in zip(self.conv_params, conv_gparams):
                conv_updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.conv_params, conv_gparams):
                inc_c = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc_c - learning_rate * gparam
                conv_updates[inc_c] = updated_inc
                conv_updates[param] = param + updated_inc


        # theano functions
        train_fn_siamese = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[cost, errors],
              updates=updates,
              givens={
                self.x: siamese_train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.x_p: siamese_train_set_x_p[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: siamese_train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        train_fn_conv = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[conv_cost, conv_errors],
              updates=conv_updates,
              givens={
                self.x: conv_train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y_: conv_train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})


        test_score_i = theano.function([index], conv_errors,
                 givens={
                   self.x: conv_test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y_: conv_test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], conv_errors,
              givens={
                 self.x: conv_valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y_: conv_valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # for now valid and test score and conv path on lisa dataset
        def valid_score():
            return [valid_score_i(i) for i in xrange(conv_valid_set_x.get_value(borrow=True).shape[0] / batch_size)]

        def test_score():
            return [test_score_i(i) for i in xrange(conv_test_set_x.get_value(borrow = True).shape[0] / batch_size)]

        return train_fn_conv, train_fn_siamese, valid_score, test_score

class SiameseVariant(object):

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
        self.y_p = tensor.ivector('y_p')


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
                            n_outs = n_outs[0],
                            act_enc = "sigmoid",
                            irange = irange,
                            bias_init = bias_init)

        self.mlp_p = DropOutMLP(input_corruptors = input_corruptors,
                            hidden_corruptors = hidden_corruptors,
                            n_units = n_units,
                            n_outs = n_outs[1],
                            act_enc = "sigmoid",
                            irange = irange,
                            bias_init = bias_init)


        if fine_tune:
            self.params = base_model.model.mlp.hiddens._params + self.mlp._params + self.mlp_p._params
        else:
            self.params = self.mlp._params + self.mlp_p._params

    def negative_log_likelihood(self, x, y):
        return -tensor.mean(tensor.log(self.mlp.p_y_given_x(x))[tensor.arange(y.shape[0]), y])

    def negative_log_likelihood_p(self, x, y):
        return -tensor.mean(tensor.log(self.mlp_p.p_y_given_x(x))[tensor.arange(y.shape[0]), y])

    def squared_error(self, x, y):
        return ((self.mlp_p.encode(x) - y) ** 2).mean()

    def errors(self, x, y):
        return (tensor.neq(self.mlp.predict_y(x), y)).mean()


    def jacobians(self, x):
        j1 = self.mlp_p.layers[0].jacobian_h_x(x)
        j2 = self.mlp.hiddens.layers[0].jacobian_h_x(x)

        #return (tensor.dot(j1[0].T, j2[0]) ** 2.).sum()
        def multi(x1, x2):
            return (tensor.dot(x1.T, x2) ** 2.).sum()

        results, _ = theano.scan(fn = multi, sequences = [j1, j2])
        return results.mean()

    def sigmoid_orthogonality(self, x):

        h1 = self.mlp.hiddens.layers[0](x)
        h2 = self.mlp_p.hiddens.layers[0](x)
        j1 = self.mlp.hiddens.layers[0].weights * (h1 * (1-h1)).dimshuffle(0, 'x', 1)
        j2 = self.mlp.hiddens.layers[0].weights * (h2 * (1-h2)).dimshuffle(0, 'x', 1)

        def multi(x1, x2):
            return (tensor.dot(x1.T, x2) ** 2.).sum()

        results, _ = theano.scan(fn = multi, sequences = [j1, j2])
        return results.mean()

    def build_finetune_functions(self, datasets, batch_size, coeffs, enable_momentum):

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
        nll_cost = self.negative_log_likelihood(self.inputs, self.y)
        nll_p_cost = self.negative_log_likelihood_p(self.inputs, self.y_p)
        mlp_l1 = tensor.abs_(self.mlp.hiddens.layers[-1].weights).mean()
        mlp_l1 += tensor.abs_(self.mlp.log_layer.W).mean()
        reg_l1 = numpy.mean([abs(item).mean() for item in self.mlp_p.weights])
        jacob_cost = self.sigmoid_orthogonality(self.inputs)

        #cost = nll_cost + coeffs['nll_p'] * sr_cost + coeffs['jacob'] * jacob_cost + coeffs['l1'] * mlp_l1 + coeffs['l1_p'] * reg_l1
        cost = nll_cost + coeffs['nll_p'] * nll_p_cost + coeffs['jacob'] * jacob_cost
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
              outputs=[nll_cost, errors],
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.x_p: train_set_x_p[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size],
                self.y_p: train_set_y_p[index * batch_size:
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
