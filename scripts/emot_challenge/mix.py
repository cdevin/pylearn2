import numpy as np

import theano
from theano import config
from theano import tensor
from theano.printing import Print
from theano.compat.python2x import OrderedDict

from pylearn2.costs.cost import Cost
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.models import Model
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

from emotiw.common.datasets.faces.afew2_facetubes import AFEW2FaceTubes
from emotiw.common.datasets.faces.facetubes import FaceTubeSpace
from emotiw.wardefar.crf_theano import forward_theano as crf


class FrameCRF(Model):
    """Logistic regression on top of mean along axis 't' of inputs"""
    def __init__(self, mlp, final_layer, n_classes = None, input_source='features', input_space=None):

        if n_classes is None:
            if hasattr(mlp.layers[-1], 'dim'):
                self.n_classes = mlp.layers[-1].dim
            elif hasattr(mlp.layers[-1], 'n_classes'):
                self.n_classes = mlp.layers[-1].n_classes
            else:
                raise ValueError("n_classes was not provided and couldn't be infered from the mlp's last layer")
        else:
            self.n_classes = n_classes

        self.mlp = mlp
        self.final_layer = final_layer
        self.input_source = input_source
        assert isinstance(input_space, FaceTubeSpace)
        self.input_space = input_space
        self.input_size = (input_space.shape[0]
                           * input_space.shape[1]
                           * input_space.num_channels)
        self.output_space = VectorSpace(dim=n_classes)
        #self.final_layer.input_space = self.mlp.layers[-1].get_output_space()

        self.W = theano.shared(np.zeros((n_classes, n_classes, n_classes),
                                        dtype=config.floatX))
        self.W.name = 'crf_w'
        self.name = 'crf'

    def fprop(self, inputs):

        # format inputs
        inputs = self.input_space.format_as(inputs, self.mlp.input_space)
        rval = self.mlp.fprop(inputs)
        rval = tensor.max(rval, axis=0)
        rval = rval.dimshuffle('x', 0)
        rval = self.final_layer.fprop(rval)
        #if self.mlp.output_space != self.detector_space:
            #rval = self.mlp.output_space.formt_as(self.detector_space)

        #import ipdb
        #ipdb.set_trace()
        #rval = crf(rval, self.W)

        return rval

    def get_params(self):
        #return self.mlp.get_params() + [self.W]
        return self.mlp.get_params() + self.final_layer.get_params()

    def get_input_source(self):
        return self.input_source

    def get_input_space(self):
        return self.input_space

    def get_monitoring_data_specs(self):
        space = CompositeSpace((self.get_input_space(),
                                VectorSpace(dim=1)))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):

        X, Y = data
        rval = OrderedDict()

        state = self.fprop(X)
        ch = self.get_monitoring_channels_from_state(state, Y)
        if not isinstance(ch, OrderedDict):
            raise TypeError(str((type(ch), str(self))))
        for key in ch:
            rval[self.name+'_'+key]  = ch[key]
        return rval

    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis = 1)
        rval = OrderedDict([
            ('mean_max_class', mx.mean()),
            ('max_max_class', mx.max()),
            ('min_max_class', mx.min())])

        if target is not None:
            y_hat = tensor.argmax(state, axis=1)
            #import ipdddb
            #ipdb.set_trace()
            y = target
            misclass = tensor.neq(y, y_hat).mean()
            misclass = tensor.cast(misclass, config.floatX)
            rval['misclass'] = misclass
        return rval

class DummyCost(Cost):
    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        X, Y = data
        # Make Y one-hot binary.

        Y = OneHotFormatter(model.n_classes).theano_expr(
                tensor.addbroadcast(Y, 1).dimshuffle(0).astype('int8'))
        #Y = tensor.alloc(Y, X.shape[0], model.n_classes)
        # Y_hat is a softmax estimate
        Y_hat = model.fprop(X)
        # Code copied from pylearn2/models/mlp.py:Softmax.cost
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, tensor.nnet.Softmax)
        z, = owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - tensor.log(tensor.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        #log_prob = Print('log_prob', attrs=('shape', 'min', 'max', '__str__'))(log_prob)
        # we use sum and not mean because this is really one variable per row
        #Y = Print('Y')(Y)
        log_prob_of = (Y * log_prob).sum(axis=1)
        #log_prob_of = Print('log_prob_of')(log_prob_of)
        assert log_prob_of.ndim == 1
        rval = log_prob_of.mean()
        return -rval

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(),
                                VectorSpace(dim=1)])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)


def test_afew2ft_train():
    n_classes = 7
    dataset = AFEW2FaceTubes(which_set='Train')
    monitoring_dataset = {
        'train': dataset,
        'valid': AFEW2FaceTubes(which_set='Val')}
    model = DummyModel(n_classes=n_classes,
                       input_space=dataset.get_data_specs()[0].components[0])
    cost = DummyCost()
    termination_criterion = EpochCounter(10)

    learning_rate = 1e-6
    batch_size = 1
    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=batch_size,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion)
    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None)
    train.main_loop()
