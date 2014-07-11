from noisylearn.projects.gating2 import vLBL
from theano import tensor as T
import theano
import numpy as np
import ipdb

model = vLBL(dict_size = 1000, dim = 100, context_length = 3,k = 2)

X = T.matrix()
y = T.vector()

x_ = np.array([[1,2,3], [4,5,6]]).astype('float32')
y_ = np.array([4, 3]).astype('float32')

X.tag.test_value = x_
y.tag.test_value = y_

X = T.cast(X, 'int32')
y = T.cast(y, 'int32')


score = model.score(X, y)
delta = model.delta((X,y))

model.get_params()
f = theano.function([X, y], [score, delta], on_unused_input='ignore', allow_input_downcast=True)



res, res2 = f(x_, y_)
ipdb.set_trace()
