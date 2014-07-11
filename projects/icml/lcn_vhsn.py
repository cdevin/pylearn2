from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN


#train = SVHN('train_all')
#del train
#valid = SVHN('valid')
#del valid
#test = SVHN('test')
#del test
#print aaa

train = SVHN('train_all', path = '/data/lisatmp2/mirzamom/data/SVHN/icpr/')
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.LeCunLCN_ICPR((32,32)))
train.apply_preprocessor(pipeline, can_fit = True)
del train

valid = SVHN('valid', path = '/data/lisatmp2/mirzamom/data/SVHN/icpr/')
valid.apply_preprocessor(pipeline, can_fit = False)
del valid

test = SVHN('test', path = '/data/lisatmp2/mirzamom/data/SVHN/icpr/')
test.apply_preprocessor(pipeline, can_fit = False)
del test




#train = SVHN('train', path = '/data/lisatmp2/mirzamom/data/SVHN/channel/')
#pipeline = preprocessing.Pipeline()
#pipeline.items.append(preprocessing.LeCunLCNChannels((32,32)))
#train.apply_preprocessor(pipeline, can_fit = True)
#del train

#valid = SVHN('valid', path = '/data/lisatmp2/mirzamom/data/SVHN/channel/')
#valid.apply_preprocessor(pipeline, can_fit = False)
#del valid

#test = SVHN('test', path = '/data/lisatmp2/mirzamom/data/SVHN/channel/')
#test.apply_preprocessor(pipeline, can_fit = False)
#del test
