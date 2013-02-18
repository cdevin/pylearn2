from pylearn2.datasets import preprocessing
from svhn import SVHN


data = SVHN('test', path = '/data/lisatmp2/mirzamom/data/SVHN/')
#data = SVHN('splited_train', path = '/data/lisatmp2/mirzamom/data/SVHN/')
#data = SVHN('valid', path = '/data/lisatmp2/mirzamom/data/SVHN/')
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.LeCunLCNChannels((32,32)))
data.apply_preprocessor(pipeline)
