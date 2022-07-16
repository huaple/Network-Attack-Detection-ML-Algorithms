
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed July 18 11:04:32 2018

@author: Ke-Hsin,Lo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import nesssary package
import math
import sys
import numpy
from numpy import Inf
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import random
import matplotlib.pyplot as plt

def load_train_data(train_ratio=1):
    data = pd.read_csv('./UNSW_NB15_training-set_selected.csv', header=None,
                       names=['x%i' % (i) for i in range(37)] + ['y'])
    Xt = numpy.asarray(data[['x%i' % (i) for i in range(37)]])
    yt = numpy.asarray(data['y'])
    return sklearn.model_selection.train_test_split(Xt, yt, test_size=1 - train_ratio, random_state=0)


def load_test_data(train_ratio=0):
    data = pd.read_csv('./UNSW_NB15_testing-set_selected.csv', header=None,
                       names=['x%i' % (i) for i in range(37)] + ['y'])
    Xtt = numpy.asarray(data[['x%i' % (i) for i in range(37)]])
    ytt = numpy.asarray(data['y'])
    return sklearn.model_selection.train_test_split(Xtt, ytt, test_size=1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train)))  # Transforms features by scaling each feature to a given range(0~1) in order to reinforce dataset and fit training set.
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

