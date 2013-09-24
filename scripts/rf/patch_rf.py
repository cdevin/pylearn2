""" This script train bunch of rf on image pathces
"""
import numpy as np
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.preprocessing import ExtractGridPatches, ReassembleGridPatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ipdb


def get_data(which_set = 'train'):

    pr = ExtractGridPatches(patch_shape = (7, 7), patch_stride = (6, 6))
    ds = MNIST(which_set, preprocessor = pr)

    print ds.X.shape, ds.y.shape
    return ds.X, ds.y



def rf_train(train_x, train_y):

    rf = RandomForestClassifier(n_estimators = 100, n_jobs = 6)
    rf.fit(train_x, train_y)


def rf_test(rf, test_x, test_y, num_path):

    y_hat = []
    #for test_x, test_y


if __name__ == "__main__":
    get_data()
