# -*- coding: utf-8 -*-
"""
Created on Thu Oct 4 17:55:20 2020

@author: Xiaoxuan Jia
"""

import numpy as np
import scipy
from scipy.stats import norm
import numpy.random as npr
import random

import utils as ut
import learningutil as lt

def d_prime(CF):
    d = []
    for i in range(len(CF[1])):
        H = CF[i, i]/sum(CF[:,i]) # H = target diagnal/target column
        tempCF = scipy.delete(CF, i, 1) # delete the target column
        F = sum(tempCF[i,:])/sum(tempCF)
        d.append(norm.ppf(H)-norm.ppf(F))
    return d


def sample_with_replacement(list):
    l = len(list) # the sample needs to be as long as list
    r = xrange(l)
    _random = random.random
    return [list[int(_random()*l)] for i in r] # using


def compute_CM(neuron, meta, obj, s, train, test):
    metric_kwargs = {'model_type': 'MCC2'} # multi-class classifier
    
    eval_config = {
        'train_q': {'obj': [obj[0], obj[1]]}, # train on all sizes
        'test_q': {'obj': [obj[0], obj[1]], 's': [s]},   #'size_range': [1.3],  
        'npc_train':  train, #smaller than total number of samples in each split_by object
        'npc_test':   test,
        'npc_validate':  0,
        'num_splits':    100,
        'split_by': 'obj',
        'metric_screen': 'classifier', # use correlation matrix as classifier 
        'labelfunc': 'obj',
        'metric_kwargs': metric_kwargs,
    }
    result = ut.compute_metric_base(neuron, meta, eval_config)

    # sum of the CMs is equal to npc_test*number of objs
    CMs = []
    for i in range(eval_config['num_splits']):
        temp = np.array(result['result_summary']['cms'])[:,:,i]
        CMs.append(lt.normalize_CM(temp))
    d =  ut.dprime(np.mean(CMs,axis=0))[0]
    return CMs, d


def compute_CM_samesize(neuron, meta, obj, s, train, test):
    metric_kwargs = {'model_type': 'MCC2'} # multi-class classifier
    
    eval_config = {
        'train_q': {'obj': [obj[0], obj[1]], 's': [s]}, # train on particular size
        'test_q': {'obj': [obj[0], obj[1]], 's': [s]},   #test on particular size 
        'npc_train':  train, #smaller than total number of samples in each split_by object
        'npc_test':   test,
        'npc_validate':  0,
        'num_splits':    100,
        'split_by': 'obj',
        'metric_screen': 'classifier', # use correlation matrix as classifier 
        'labelfunc': 'obj',
        'metric_kwargs': metric_kwargs,
    }
    result = ut.compute_metric_base(neuron, meta, eval_config)

    # sum of the CMs is equal to npc_test*number of objs
    CMs = []
    for i in range(eval_config['num_splits']):
        temp = np.array(result['result_summary']['cms'])[:,:,i]
        CMs.append(lt.normalize_CM(temp))
    d =  ut.dprime(np.mean(CMs,axis=0))[0]
    return CMs, d


def compute_CM_fixed_classifier(neuron, meta, obj, s, train, test):
    metric_kwargs = {'model_type': 'MCC2'} # multi-class classifier
    
    eval_config = {
        'train_q': {'obj': [obj[0], obj[1]], 'test_phase':['Pre']}, # train on all sizes
        'test_q': {'obj': [obj[0], obj[1]], 's': [s], 'test_phase':['Post']},   #'size_range': [1.3],  
        'npc_train':  train, #smaller than total number of samples in each split_by object
        'npc_test':   test,
        'npc_validate':  0,
        'num_splits':    100,
        'split_by': 'obj',
        'metric_screen': 'classifier', # use correlation matrix as classifier 
        'labelfunc': 'obj',
        'metric_kwargs': metric_kwargs,
    }
    result = ut.compute_metric_base(neuron, meta, eval_config)

    # sum of the CMs is equal to npc_test*number of objs
    CMs = []
    for i in range(eval_config['num_splits']):
        temp = np.array(result['result_summary']['cms'])[:,:,i]
        CMs.append(lt.normalize_CM(temp))
    d =  ut.dprime(np.mean(CMs,axis=0))[0]
    return CMs, d