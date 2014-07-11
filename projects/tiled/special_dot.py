"""
Author: Razvan Pascanu
"""


import numpy
import theano
import theano.tensor as TT
from theano import tensor
from theano.gradient import DisconnectedType
from theano.sandbox import cuda
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from utils import softmax

class GroupDot(theano.gof.Op):
    def __init__(self, n_groups, gpu):
        """
        Computes only the forward pass when doing the class like structure
        that Tomas proposed to speed up the output layer (which contains
        many softmax units)
        """
        self.n_groups = n_groups
        self.gpu = gpu


    def __eq__(self, other):
        return type(self) == type(other) and \
                self.n_groups == other.n_groups and \
                self.gpu == other.gpu

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups) ^ hash(self.gpu)

    def make_node(self, vec, mat, bias, index):
        if self.gpu:
            if isinstance(vec.type, TT.TensorType):
                vec = cuda.gpu_from_host(vec)
            if isinstance(mat.type, TT.TensorType):
                mat = cuda.gpu_from_host(mat)
            if isinstance(bias.type, TT.TensorType):
                bias = cuda.gpu_from_host(bias)
        else:
            vec  = TT.as_tensor_variable(vec)
            mat  = TT.as_tensor_variable(mat)
            bias = TT.as_tensor_variable(bias)

        index = TT.as_tensor_variable(index)
        assert vec.ndim == 2
        assert mat.ndim == 3
        assert bias.ndim == 2
        assert index.ndim == 1
        assert 'int' in index.dtype
        return theano.gof.Apply(self,
                                [vec, mat, bias, index],
                                [vec.type()])


    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        if self.gpu:
            shared = theano.shared
        else:
            shared = TT._shared
        self.W = shared(numpy.zeros((2,2), dtype='float32'))

        self.b = shared(numpy.zeros((2,), dtype='float32'))
        self.h = shared(numpy.zeros((2,2), dtype='float32'))
        self.out = shared(numpy.zeros((2,2), dtype='float32'))
        out = TT.dot(self.h, self.W) + self.b
        updates  = OrderedDict({self.out:out})
        self.step = theano.function(
            [],
            [],
            name='step',
            updates=updates)

        p = self.execute
        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            return r
        self.tmp_h = None
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval


    def execute(self, node, ins, _outs):
        state_below, matrix, biases, groups = ins

        #if not hasattr(self, 'precompute') or self.tmp_h.shape[0] < state_below.shape[0]:
        if not hasattr(self, 'precompute') or (self.tmp_h is not None and self.tmp_h.shape[0] < state_below.shape[0]):
            if self.gpu:
                if self.tmp_h:
                    del self.tmp_h
                self.tmp_h = cuda.CudaNdarray.zeros(state_below.shape)
            self.precompute = True
        if _outs[0][0] and _outs[0][0].shape == state_below.shape:
            pass
        else:
            nw_shape = (state_below.shape[0], biases.shape[1])
            if self.gpu:
                _outs[0][0] = cuda.CudaNdarray.zeros(nw_shape)
            else:
                _outs[0][0] = numpy.zeros(nw_shape, dtype=state_below.dtype)
        for pos in xrange(self.n_groups):
            mask = groups == pos
            if mask.sum() != 0:
                self.W.set_value(matrix[pos], borrow=True)
                self.b.set_value(biases[pos], borrow=True)
                if self.gpu:
                    pdx = 0
                    for jdx in xrange(groups.shape[0]):
                        if groups[jdx] == pos:
                            self.tmp_h[pdx] = state_below[jdx]
                            pdx += 1
                    self.h.set_value(self.tmp_h[:pdx], borrow=True)
                else:
                    self.h.set_value(state_below[mask],
                                     borrow=True)
                self.step()
                values = self.out.get_value(borrow=True,
                                            return_internal_type=True)
                if self.gpu:
                    pdx = 0
                    for jdx in xrange(groups.shape[0]):
                        if groups[jdx] == pos:
                            _outs[0][0][jdx] = values[pdx]
                            pdx += 1
                else:
                    _outs[0][0][mask] = values

    def grad(self, inputs, grads):
        state_below, matrix, biases, groups = inputs
        gout, = grads
        rval = GradGroupDot(n_groups = self.n_groups,
                             gpu = self.gpu)(state_below, matrix, biases,
                                             groups, gout)
        return rval + [DisconnectedType()()]


class GradGroupDot(theano.gof.Op):
    def __init__(self, n_groups, gpu):
        """
        Computes only the forward pass when doing the class like structure
        that Tomas proposed to speed up the output layer (which contains
        many softmax units)
        """
        self.n_groups = n_groups
        self.gpu = gpu


    def __eq__(self, other):
        return type(self) == type(other) and \
                self.n_groups == other.n_groups and \
                self.gpu == other.gpu

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups) ^ hash(self.gpu)

    def make_node(self, vec, mat, bias, index, grad_on_out):
        if self.gpu:
            if isinstance(vec.type, TT.TensorType):
                vec = cuda.gpu_from_host(vec)
            if isinstance(mat.type, TT.TensorType):
                mat = cuda.gpu_from_host(mat)
            if isinstance(bias.type, TT.TensorType):
                bias = cuda.gpu_from_host(bias)
            if isinstance(grad_on_out.type, TT.TensorType):
                grad_on_out = cuda.gpu_from_host(grad_on_out)
        else:
            vec  = TT.as_tensor_variable(vec)
            mat  = TT.as_tensor_variable(mat)
            bias = TT.as_tensor_variable(bias)
            grad_on_out = TT.as_tensor_variable(grad_on_out)

        index = TT.as_tensor_variable(index)
        assert vec.ndim == 2
        assert mat.ndim == 3
        assert bias.ndim == 2
        assert index.ndim == 1
        assert 'int' in index.dtype
        return theano.gof.Apply(self,
                                [vec, mat, bias, index, grad_on_out],
                                [vec.type(), mat.type(), bias.type()])


    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        if self.gpu:
            shared = theano.shared
        else:
            shared = TT._shared
        self.W = shared(numpy.zeros((2,2), dtype='float32'))

        self.b = shared(numpy.zeros((2,), dtype='float32'))
        self.h = shared(numpy.zeros((2,2), dtype='float32'))
        #self.out = shared(numpy.zeros((2,2), dtype='float32'))
        self.grad_on_out = shared(numpy.zeros((2,2), dtype='float32'))
        self.gW = shared(numpy.zeros((2,2), dtype='float32'))
        self.gh = shared(numpy.zeros((2,2), dtype='float32'))
        self.gb = shared(numpy.zeros((2,), dtype='float32'))

        gW = TT.dot(self.h.T, self.grad_on_out)
        gh = TT.dot(self.grad_on_out, self.W.T)
        gb = self.grad_on_out.sum(0)

        updates  = OrderedDict({self.gW:gW,
                                self.gb:gb,
                                self.gh:gh})
        self.step = theano.function([],[],
                                    updates = updates,
                                    name='grad_step')

        self.tmp_h = None
        self.tmp_grad_out = None
        p = self.execute
        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            return r
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval


    def execute(self, node, ins, _outs):
        state_below, matrix, biases, groups, grad_on_out = ins
        #if not hasattr(self, 'precompute') or self.tmp_h.shape[0] < state_below.shape[0]:
        if not hasattr(self, 'precompute') or (self.tmp_h is not None and self.tmp_h.shape[0] < state_below.shape[0]):
            if self.gpu:
                if self.tmp_h:
                    del self.tmp_h
                if self.tmp_grad_out:
                    del self.tmp_grad_out
                self.tmp_h = cuda.CudaNdarray.zeros(state_below.shape)
                self.tmp_grad_out = cuda.CudaNdarray.zeros(grad_on_out.shape)
            self.precompute = True

        if _outs[0][0] and _outs[0][0].shape == state_below.shape:
            pass
        else:
            if self.gpu:
                _outs[0][0] = cuda.CudaNdarray.zeros(state_below.shape)
            else:
                _outs[0][0] = numpy.zeros_like(state_below)

        if _outs[1][0] and _outs[1][0].shape == matrix.shape:
            pass
        else:
            if self.gpu:
                _outs[1][0] = cuda.CudaNdarray.zeros(matrix.shape)
            else:
                _outs[1][0] = numpy.zeros_like(matrix)

        if _outs[2][0] and _outs[2][0].shape == biases.shape:
            pass
        else:
            if self.gpu:
                _outs[2][0] = cuda.CudaNdarray.zeros(biases.shape)
            else:
                _outs[2][0] = numpy.zeros_like(biases)


        for pos in xrange(self.n_groups):
            mask = groups == pos
            if mask.sum() != 0:
                self.W.set_value(matrix[pos], borrow=True)
                self.b.set_value(biases[pos], borrow=True)


                if self.gpu:
                    pdx = 0
                    for jdx in xrange(groups.shape[0]):
                        if groups[jdx] == pos:
                            self.tmp_h[pdx] = state_below[jdx]
                            self.tmp_grad_out[pdx]= grad_on_out[jdx]
                            pdx += 1
                    self.h.set_value(self.tmp_h[:pdx], borrow=True)
                    self.grad_on_out.set_value(self.tmp_grad_out[:pdx],
                                               borrow=True)
                else:
                    self.h.set_value(state_below[mask],
                                     borrow=True)
                    self.grad_on_out.set_value(grad_on_out[mask],
                                               borrow=True)
                self.step()
                gh = self.gh.get_value(borrow=True,
                                       return_internal_type=True)
                gW = self.gW.get_value(borrow=True,
                                       return_internal_type=True)
                gb = self.gb.get_value(borrow=True,
                                       return_internal_type=True)
                if self.gpu:
                    pdx = 0
                    for jdx in xrange(groups.shape[0]):
                        if groups[jdx] == pos:
                            _outs[0][0][jdx] = gh[pdx]
                            pdx += 1
                else:
                    _outs[0][0][mask] = gh
                _outs[1][0][pos] += gW
                _outs[2][0][pos] += gb

    def grad(self, inputs, grads):
        raise NotImplemented


