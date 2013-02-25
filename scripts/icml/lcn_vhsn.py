from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN


#train = SVHN('train')
#del train
#valid = SVHN('valid')
#del valid
#test = SVHN('test')
#del test
#print aaa

train = SVHN('train', path = '/data/lisatmp2/mirzamom/data/SVHN/')
pipeline = preprocessing.Pipeline()
#pipeline.items.append(preprocessing.LeCunLCNChannelsPyTables((32,32)))
pipeline.items.append(preprocessing.LeCunLCN_ICPR((32,32)))
train.apply_preprocessor(pipeline, can_fit = True)
del train

valid = SVHN('valid', path = '/data/lisatmp2/mirzamom/data/SVHN/')
valid.apply_preprocessor(pipeline, can_fit = False)
del valid

test = SVHN('test', path = '/data/lisatmp2/mirzamom/data/SVHN/')
test.apply_preprocessor(pipeline, can_fit = False)
del valid
