from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano import sparse as S
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models import mlp
from theano.gof.op import get_debug_values

from pylearn2.models.mlp import Layer, MLP, Linear
from pylearn2.space import Space
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.utils import sharedX
from theano.sandbox.cuda.blocksparse import (sparse_block_dot_SS,
                                             sparse_block_gemv_ss)
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer
from pylearn2.costs.mlp import Default
from itertools import izip
from pylearn2.sandbox.nlp.models.mlp import Softmax
from theano.tensor.sort import argsort

class ProjectionLayerGater(ProjectionLayer):
    def __init__(self, dim,num_blocks,block_size,layer_name,gater=None,irange=None,istdev=None):
        self.dim = dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        super(ProjectionLayer, self).__init__()
        self.layer_name = layer_name
        self.gater = gater
        if irange is None and istdev is None:
            raise ValueError("ProjectionLayer needs either irange or"
                             "istdev in order to intitalize the projections.")
        elif irange is not None and istdev is not None:
            raise ValueError("ProjectionLayer was passed both irange and "
                             "istdev but needs only one")
        else:
            self._irange = irange
            self._istdev = istdev

    def fprop(self,state_below,targets=None):
        z = self.transformer.project(state_below)
        return z

class MLP_GatedLinear(Linear):
    def __init__(self,num_blocks,block_size,layer_name,gater=None,
                 irange=None,istdev=None,sparse_init=None,sparse_stdev=1.,include_prob=1.0,init_bias=0.,
                 W_lr_scale=None,b_lr_scale=None,mask_weights=None,max_row_norm=None,max_col_norm=None,
                 min_col_norm=None,softmax_columns=None,copy_input=None,use_abs_loss=False,use_bias=True):
        dim = num_blocks * block_size
        self.num_blocks = num_blocks
        self.block_size = block_size
        super(MLP_GatedLinear, self).__init__(
                 dim,layer_name,
                 irange=None,istdev=None,sparse_init=None,sparse_stdev=1.,include_prob=1.0,init_bias=0.,
                 W_lr_scale=None,b_lr_scale=None,mask_weights=None,max_row_norm=None,max_col_norm=None,
                 min_col_norm=None,softmax_columns=None,copy_input=None,use_abs_loss=False,use_bias=True)
        self.__dict__.update(locals())
        del self.self

    def get_params(self):
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        if self.use_bias:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval + self.gater.get_params()


    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        W ,= self.transformer.get_params()
        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        #var = self.gater.layers[-1].W
        #var = T.argmax(var, axis=1).std().astype(theano.config.floatX)
        rval = OrderedDict([
                           ('row_norms_min'  , row_norms.min()),
                           ('row_norms_mean' , row_norms.mean()),
                           ('row_norms_max'  , row_norms.max()),
                           ('col_norms_min'  , col_norms.min()),
                           ('col_norms_mean' , col_norms.mean()),
                           ('col_norms_max'  , col_norms.max()),
         #                  ('softmax_weights_std', var), 
                           ])
        if (state is not None) or (state_below is not None):
            if state is None:
                gate = self.gater.fprop(state_below)
                rval['gaterout'] = len(np.where(gate==1)[0])
            else:
                pass
        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)
        
        gate = self.gater.fprop(state_below)
        gate = T.extra_ops.repeat(gate,self.block_size,axis=1)
        z = self.transformer.lmul(state_below) + self.b
        return z*gate

class MLP_GatedLinear_Op(Linear):

    def __init__(self,num_blocks,block_size,layer_name,gater=None,
                 irange=None,istdev=None,sparse_init=None,sparse_stdev=1.,include_prob=1.0,init_bias=0.,
                 W_lr_scale=None,b_lr_scale=None,mask_weights=None,max_row_norm=None,max_col_norm=None,
                 min_col_norm=None,softmax_columns=None,copy_input=None,use_abs_loss=False,use_bias=True):
        dim = num_blocks * block_size
        self.num_blocks = num_blocks
        self.block_size = block_size
        super(MLP_GatedLinear_Op, self).__init__(
                 dim,layer_name,
                 irange=None,istdev=None,sparse_init=None,sparse_stdev=1.,include_prob=1.0,init_bias=0.,
                 W_lr_scale=None,b_lr_scale=None,mask_weights=None,max_row_norm=None,max_col_norm=None,
                 min_col_norm=None,softmax_columns=None,copy_input=None,use_abs_loss=False,use_bias=True)
        self.__dict__.update(locals())
        del self.self

    def get_params(self):
        #W, = self.transformer.get_params()
        #assert W.name is not None
        #rval = self.transformer.get_params()
        #assert not isinstance(rval, set)
        
        #rval = list(rval)
        rval = []
        rval.append(self.W)
        if self.use_bias:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval + self.gater.get_params()

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)
        self.output_space = VectorSpace(self.dim)
        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.input_dim, self.dim) * self.istdev
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))

            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        self.transformer = MatrixMul(W)

        mlp = self.get_mlp()
        fromlayer = -1
        for i in xrange(1,len(mlp.layers)-1):
            if self == mlp.layers[i]:
                fromlayer = i-1
                break
        
        if fromlayer !=-1:
            newshape = (mlp.layers[fromlayer].num_blocks,self.num_blocks,mlp.layers[fromlayer].block_size,self.block_size)
            W =np.reshape(W,newshape)
        
        W = sharedX(W)
        W.name = self.layer_name+'_W'
        self.W = W
        
        if self.mask_weights is not None:
            expected_shape = (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " +
                                 str(expected_shape)+" but got " +
                                 str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        W = self.W

        #assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=3))
        col_norms = T.sqrt(sq_W.sum(axis=2))

        rval = OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                          #  ('softmax_weights_std', var), 
                            ])
        
        if (state is not None) or (state_below is not None):
            if state is None:
                pass

        return rval

    def fprop(self, state_below):

        #self.input_space.validate(state_below)

        #if self.requires_reformat:
        #    state_below = self.input_space.format_as(state_below, self.desired_space)
        x = state_below
        mlp = self.get_mlp()
        for i in xrange(0,len(mlp.layers)):
            if self==mlp.layers[i]:
                curlayeridx = i        
        gate = self.gater.fprop(state_below)
        self.gater_val = argsort(gate).flatten()[-10:]
        #x = T.reshape(x,(mlp.layers[curlayeridx-1].num_blocks,mlp.layers[curlayeridx-1].block_size))
        if curlayeridx==1:
            x = T.reshape(x,(mlp.layers[curlayeridx-1].num_blocks,mlp.layers[curlayeridx-1].block_size))
        else:
            x = T.reshape(x,(10,mlp.layers[curlayeridx-1].block_size))
        
        ###         
        if curlayeridx==0:
            #with this layer block size and num blocks, make all 1s
            inIdx = T.arange(mlp.layers[curlayeridx-1].num_blocks)
            outIdx = self.gater_val.flatten()
        elif curlayeridx==len(mlp.layers)-1:
            inIdx = mlp.layers[curlayeridx-1].gater_val.flatten()
            #x = x[inIdx]
            outIdx = T.arange(self.num_blocks)#flatnonzero(T.ones_like(mlp.layers[curlayeridx].gater_val))
        else:
            inIdx = mlp.layers[curlayeridx-1].gater_val.flatten()
            #x = x[inIdx]
            outIdx = self.gater_val.flatten()

        self.inIdx = inIdx    
        self.outIdx = outIdx 
             
        B = T.reshape(self.b,(self.num_blocks,self.block_size))
        z = sparse_block_dot_SS(self.W,x,self.inIdx,B,self.outIdx)
        #rval = T.alloc(0.,(self.num_blocks*self.block_size)).reshape((self.num_blocks,self.block_size))
        #rval = T.set_subtensor(rval[self.outIdx],z).flatten().dimshuffle('x',0)
        return z
        



class Cost(Default):
    def get_gradients(self, model, data, ** kwargs):

        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError, e:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling " + str(type(self)) + ".expr"
            logger.error(type(self))
            logger.error(e.message)
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        params = list(model.get_params())
        a = [model.layers[i].inIdx for i in xrange(1,len(model.layers)-1)]
        b = [model.layers[i].outIdx for i in xrange(1,len(model.layers)-1)]
        a = a + b
        grads = T.grad(cost, params, disconnected_inputs='ignore', consider_constant=a)

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates


class Softmax_flat(Softmax):
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            b = self.b
            state_below = state_below.flatten()
            Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval
