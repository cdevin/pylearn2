from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN
import gc

# The SVHN class is desinged in a way that does not allowed any change to original
# h5 file. So we first create he h5 files then copu them to some temp folder for any change


#train = SVHN('train_all')
#del train
#gc.collect()
#train = SVHN('splitted_train', path = '/Tmp/mirzamom/data/SVHN/')
#del train
#gc.collect()
#valid = SVHN('valid')
#del valid
#gc.collect()
#test = SVHN('test')
#del test
#gc.collect()
#
### Manually copy data tp local_path
##path = '${LOCAL_SVHN}'
#path = '/data/lisatmp2/mirzamom/data/SVHN/icpr/'
path = '/Tmp/mirzamom/data/SVHN/'

train = SVHN('splitted_train', path = path)
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalizationPyTables())
pipeline.items.append(preprocessing.LeCunLCN((32,32)))
train.apply_preprocessor(pipeline, can_fit = True)
del train
gc.collect()

valid = SVHN('valid', path = path)
valid.apply_preprocessor(pipeline, can_fit = False)
del valid

test = SVHN('test', path = path)
test.apply_preprocessor(pipeline, can_fit = False)
del test


