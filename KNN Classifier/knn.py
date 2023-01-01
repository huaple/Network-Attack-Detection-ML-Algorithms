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
        sum_xx += math.pow(float(v1[i]), 2)
        sum_xy += float(v1[i]) * float(v2[i])
        sum_yy += math.pow(float(v2[i]), 2)

    return sum_xy / math.sqrt(sum_xx * sum_yy)

def cosine_distance(v1, v2):
    1-cosine_similarity(v1,v2)

# KNN prediction and model training
def knn_predict(test_data, train_data, k_value, category):
    totalcount = 0
    for i in test_data: #select tested data
        cos_similarity_list = [] # all distance array

        classNum=dict() #a dictionary of nebor
        classNum['Normal'] = 0
        classNum['Reconnaissance'] = 0
        classNum['Exploits'] = 0
        classNum['Fuzzers'] = 0
        classNum['DoS'] = 0
        classNum['Generic'] = 0
        classNum['Shellcode'] = 0
        classNum['Analysis'] = 0
        classNum['Worms'] = 0
        classNum['Backdoors'] = 0

        jcount = 0

        for j in train_data: # find in train data to get the nearest point
       #     print "i: %s" %(i)
            cos_sm = cosine_similarity(i, j)  #  1 test data  train set
            cos_similarity_list.append((category[jcount], cos_sm)) #the distance with the category
#            print cos_similarity_list # just for debugging and observing; in general running, this line will not be used.
            print "count: %s" %(jcount)
            cos_similarity_list.sort(key=operator.itemgetter(1), reverse=True) #use cos piority
            ''' similarity priority list has been built; we can find the first k nearest neighbors '''
            jcount += 1
            totalcount += 1
            print "Processing: %s" % (totalcount)

        knn = cos_similarity_list[:k_value]  # select first k neighbors

        print knn
        for k in knn: #k[0] is the most simliar.
            if k[0] == 'Normal':
                classNum['Normal'] += 1
            elif k[0] == 'Reconnaissance':
                classNum['Reconnaissance'] += 1
            elif k[0] == 'Exploits':
                classNum['Exploits'] += 1
            elif k[0] == 'Fuzzers':
                classNum['Fuzzers'] += 1
            elif k[0] == 'DoS':
                classNum['DoS'] += 1
            elif k[0] == 'Generic':
                classNum['Generic'] += 1
            elif k[0] == 'Shellcode':
                classNum['Shellcode'] += 1
            elif k[0] == 'Analysis':
                classNum['Analysis'] += 1
            elif k[0] == 'Worms':
                classNum['Worms'] += 1
            elif k[0] == 'Backdoors':
                classNum['Backdoors'] += 1
