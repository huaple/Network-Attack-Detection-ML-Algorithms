#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 18:13:26 2018

@author: Ke-Hsin, Lo
"""

import unicodecsv
import random
import operator
import math
import numpy

import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt

# getdata() function definition
def getdata(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.reader(f)
        return list(reader)





def cosine_similarity(v1, v2):

    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
 #   print "len: %d" %(len(v1))
    for i in range(0, len(v1)-1):
 #       print (v1[i])
    