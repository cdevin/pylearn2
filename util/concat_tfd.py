import pickle
import numpy


"""Concatenate all the unlabled TFD data to a single pickle file"""

paths = ["/RQexec/mirzameh/data/TFD/unsupervised/TFD_unsupervised_train_unlabeled{}.pkl".format(ind) for ind in xrange(12)]


data_x, data_y = [], []
for item in paths:
    x, y = pickle.load(open(item, 'r'))
    data_x.append(x)
    data_y.append(y)

data_x = numpy.concatenate(data_x)
data_y = numpy.concatenate(data_y)

with open("/RQexec/mirzameh/data/TFD/unsupervised/TFD_unsupervised_train_unlabeled_all.pkl", "wb") as out:
    pickle.dump([data_x, data_y], out)
