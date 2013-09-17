"""
This moduel is just to compare the perforamnce of random forest and
the features from random forest visitin nodes

"""
import numpy as np
from pylearn2.datasets.mnist import MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
import ipdb

def get_data(which_set = 'train'):

    ds = MNIST(which_set)
    return ds.X, ds.y

def rf_classify(train_x, train_y, test_x, test_y):

    rf = RandomForestClassifier(n_estimators=10, n_jobs = 6)
    rf.fit(train_x, train_y)

    y_hat = rf.predict(test_x)
    return accuracy_score(test_y, y_hat)


def get_tree_visit(tree, x, feat_size):

    node_id = 0
    _TREE_LEAF = -1
    #feaure = np.zeros(tree.max_depth)
    feature = np.zeros(feat_size)
    count = 0
    #ipdb.set_trace()
    while tree.children_left[node_id] != _TREE_LEAF:
        if x[tree.feature[node_id]] <= tree.threshold[node_id]:
            val = 1
            node_id = tree.children_left[node_id]
        else:
            val = -1
            node_id = tree.children_right[node_id]
        feature[count] = val
        count += 1

    return feature


def get_forest_visit(forest, x, max_size = 400):

    rval = []
    for est in forest.estimators_:
        rval.append(get_tree_visit(est.tree_, x, max_size))

    return np.concatenate(rval)


def test_classic():
    print rf_classify()
    #Result was 100: 0.9702, 1000: 0.9715


def test_feature():

    train_x, train_y = get_data('train')
    rf = RandomForestClassifier(n_estimators=10, n_jobs = 6)
    rf.fit(train_x, train_y)

    test_x, test_y = get_data('test')

    test_feats = []
    train_feats = []
    for x in test_x:
        test_feats.append(get_forest_visit(rf, x))

    for x in train_x:
        train_feats.append(get_forest_visit(rf, x))

    train_feats = np.array(train_feats)
    test_feats = np.array(test_feats)

    # 0.9489
    return (train_x, train_y, test_x, test_y)

def svm(train_x, train_y, test_x, test_y, C=100):
    # train svm
    clf = SVC(C = C)
    clf = LinearSVC()
    clf.fit(train_x, train_y)
    y_hat = clf.predict(test_x)
    return accuracy_score(test_y, y_hat)


if __name__ == "__main__":
    feats = test_feature()
    print rf_classify(*feats)
    print svm(*feats)
    print svm(*feats, C=10000)
    # svm rbf: 0.875
    # linear svc, c=100: 0.9182
    # linear svc, c=10000: 0.9172
