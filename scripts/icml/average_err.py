from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import numpy
import theano.tensor as T
from theano import function

_, model_path, which = sys.argv

if which not in ['g', 'a']:
    raise ValueError("Unknown option")

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = model.force_batch_size
# handle bug in older pkl files, where set_batch_size had updated
# batch_size but not force_batch_size
if hasattr(model, 'batch_size') and model.batch_size != model.force_batch_size:
    batch_size = model.batch_size


assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()


Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'
yb = model.get_output_space().make_batch_theano()
yb.name = 'yb'

if which == 'g':
    print 'geo'
    ymf = model.fprop_below(Xb, apply_dropout = True)
else:
    print 'ari'
    ymf = model.fprop(Xb, apply_dropout = True)
ymf.name = 'ymf'


batch_y = function([Xb],[ymf])


batch_size = 100
def predict():
    outs = []
    for i in xrange(test.X.shape[0]/batch_size):
        #print i
        x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        outs.append( batch_y(x_arg)[0])
    return numpy.concatenate(outs)

def accs(data):
    y_pred = numpy.argmax(data, 1)
    y = numpy.argmax(test.y, 1)
    return numpy.sum(y_pred != y) / float(y.shape[0])

def softmax(data):
    data = data - data.max()
    return numpy.exp(data) / numpy.exp(data).sum(1)[:, numpy.newaxis]


def save(data, num):
    name = "/data/lisatmp/mirzamom/results/icml2013/boost/{}_{}.npy".format(num, which)
    numpy.save(name, data)


def boost(num):
    result =[]
    for i in xrange(num):
        result.append(predict()[numpy.newaxis,:,:])

    result = numpy.concatenate(result)
    result = result.mean(0)
    save(result, num)
    if which == 'g':
        result = softmax(result)

    print "# of samples: {}, error: {}".format(num,  accs(result))

#boost(1)
#boost(2)
#boost(3)
#boost(4)

boost(1)
boost(5)
boost(10)
boost(50)
boost(100)
boost(500)
boost(1000)
boost(5000)
#boost(10000)

