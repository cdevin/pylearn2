from noisy_encoder.models.conv_pylearn import Conv, LeNet
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from theano import tensor as T

def test_get_weights():
    #Tests that the RBM, when constructed
    #with nvis and nhid arguments, supports the
    #weights interface

    model = Conv(irange = 0.05, image_shape = [32, 32], kernel_shape = [5,5],
            nchannels_input = 3, nchannels_output = 20, pool_shape = [20, 20], batch_size = 10,
            act_enc = "sigmoid")
    W = model.get_weights()

def test_get_input_space():
    #Tests that the RBM supports
    #the Space interface

    model = RBM(nvis = 2, nhid = 3)
    space = model.get_input_space()

def test_encode():
    model = Conv(irange = 0.05,  image_shape = [32, 32], kernel_shape = [5,5],
            nchannels_input = 3, nchannels_output = 20, pool_shape = [20, 20], batch_size = 10,
            act_enc = "sigmoid")
    theano_rng = RandomStreams(17)

    X = T.tensor4()

    Y = model(X)


def test_lenet_encode():
    model = LeNet(image_shape = [32, 32], kernel_shapes = ([5,5], [3,3]),
            nchannels = [3, 20, 50], pool_shape = ([2, 2],[2, 2]), batch_size = 10,
            mlp_input_corruptors = [None, None], mlp_hidden_corruptors = [None, None],
            mlp_nunits = [1000, 2000], n_outs = 20,
            act_enc = "sigmoid")
    theano_rng = RandomStreams(17)

    X = T.tensor4()

    Y = model(X)


if __name__ == "__main__":
    test_get_weights()
    test_encode()
    test_lenet_encode()
