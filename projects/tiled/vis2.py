#!/usr/bin/env python
__author__ = "Ian Goodfellow"
"""
Shows a visualization of how strong each of the words are connected to each
point in the square topology layer.

Usage: vis.py model.pkl

"""

import sys

import numpy as np

_, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

from pylearn2.models.mlp import MLP

if not isinstance(model, MLP):
    raise TypeError("This visualization script only works on MLPs.")

layers = model.layers

if len(layers) < 3:
    raise ValueError("Expected at least 3 layers: EmbeddingLinear, Linear, "
            "and SpaceConverter.")

embed, reproj, reshape = layers[0:3]

from noisylearn.projects.tiled import EmbeddingLinear
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import SpaceConverter

##########  Hack the EmbeddingLinear   #####################################

if not isinstance(embed, EmbeddingLinear):
    raise ValueError("Expected first layer to be EmbeddingLinear.")

# We now use just two dummy words for computing the norm of the second layer:
# Word 0 maps to an embedding of all 0s
# Word 1 maps to an embedding of all 1s

# Set embed to operate on only two words
embed.dict_dim = 2
# Regenerate params with new size
embed.set_input_space(embed.input_space)

# Set word 0 to all 0s and word 1 to all 1s
W, = embed.transformer.get_params()
new_val = 0. * W.get_value()
assert new_val.shape[0] == 2
new_val[1, :] = 1.
W.set_value(new_val)
b = embed.b
b.set_value(0. * b.get_value())

###### Hack the Linear ########################################################

if not isinstance(embed, Linear):
    raise ValueError("Expected second layer to be Linear reprojection.")

W, = reproj.transformer.get_params()
W.set_value(W.get_value() ** 2.)
b = reproj.b
b.set_value(b.get_value() * 0.)

###### Check the SpaceConverter ###############################################

assert isinstance(reshape, SpaceConverter)

from pylearn2.space import Conv2DSpace

space = reshape.output_space

assert isinstance(space, Conv2DSpace)

assert space.num_channels == 1

##### Compile the theano function #############################################

ipt = model.get_input_space().make_batch_theano()
embeddings = embed.fprop(ipt)
topo = reproj.fprop(embeddings)
square = reshape.fprop(topo)
for_viewer = square.dimshuffle([space.axes.index(elem) for elem in \
        ['b', 0, 1, 'c']])

from pylearn2.utils import function

f = function([ipt], for_viewer)

#### Make the heatmaps #########################################################

num_words = model.get_input_space().dim
batch = np.identity(num_words).astype(ipt.dtype)
batch = f(batch)

# These should all be squared norms
assert batch.min() >= 0.
batch = np.sqrt(batch)
tot = batch.sum(axis=0)
for i in xrange(batch.shape[0]):
    batch[i, :, :, :] /= tot
assert tot.min() > 0.
batch /= batch.max()
batch = batch * 2.0 - 1.0

from pylearn2.gui.patch_viewer import make_viewer

pv = make_viewer(batch, rescale=False)

pv.show()
