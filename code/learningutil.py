# -*- coding: utf-8 -*-
"""
Created on Thu Oct 4 16:39:50 2013

@author: Xiaoxuan Jia
"""

import json
import csv
import re
import scipy.io
import scipy.stats
import random
import numpy as np
import os
import itertools
import cPickle as pk
import pymongo
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

def SBcorrection(corr, mult_factor):
    pred = (mult_factor*corr)/(1+(mult_factor-1)*corr)
    return pred

def normalize_CM(CF):
    new_CF = np.zeros(np.shape(CF))
    for col in range(0, np.shape(CF)[1]):
        total = np.sum(CF[:,col])
        norm_col = CF[:,col]/float(total)
        new_CF[:,col] = norm_col
    return new_CF

def d_prime2x2(CF):
    H = CF[0,0]/(CF[0,0]+CF[1,0]) # H = hit/(hit+miss)
    F = CF[0,1]/(CF[0,1]+CF[1,1]) # F =  False alarm/(false alarm+correct rejection)
    if H == 1:
        H = 1-1/(2*(CF[0,0]+CF[1,0]))
    if H == 0:
        H = 0+1/(2*(CF[0,0]+CF[1,0]))
    if F == 0:
        F = 0+1/(2*(CF[0,1]+CF[1,1]))
    if F == 1:
        F = 1-1/(2*(CF[0,1]+CF[1,1]))
    d = norm.ppf(H)-norm.ppf(F)
    return d

def d_prime(CF): #have problem when called by module name, artificially change to n by 5 matrix
    d = []
    for i in range(len(CF[0][1])):
        H = CF[0][i, i]/sum(CF[0][:,i]) # H = target diagnal/target column
        tempCF = scipy.delete(CF[0], i, 1) # delete the target column
        F = sum(tempCF[i,:])/sum(tempCF)
        #if H == 1:
         #   H = 1-1/(2*sum(CF[0][:,i]))
        #if H == 0:
          #  H = 0+1/(2*sum(CF[0][:,i]))
        #if F == 0:
         #   F = 0+1/(2*sum(tempCF))
        #if F == 1:
         #   F = 1-1/(2*sum(tempCF))
        d.append(norm.ppf(H)-norm.ppf(F))
    return d

def offDmass(CF):
    return sum(CF[np.eye(CF.shape[0])==0]/float(sum(CF)))

class expDataDB(object):
    
    def __init__(self, collection, selector, numObjs, obj, trialNum):


        conn = pymongo.Connection(port = 22334, host = 'localhost')
        db = conn.mturk
        col = db[collection]
        
        self.obj = obj
        self.trialNum = trialNum
        self.subj_data = list(col.find(selector))
        self.numObjs = numObjs

        if obj != 'face':
            obj_inds = []
            for idx, t in enumerate(self.subj_data[0]['ImgData']):
                if len(np.unique(obj_inds)) == self.numObjs:
                    break
                else:
                    if len(t)<10:
                        obj_inds.append(t[0]['obj'])
                    else:
                        obj_inds.append(t['obj'])

            self.models = np.unique(obj_inds)

            self.models_idxs = {}
            for idx, model in enumerate(self.models):
                self.models_idxs[model] = idx
                self.models_idxs = self.models_idxs


        self.trial_data = self.preprocess(self.subj_data, self.obj, self.trialNum)
        self.numResp = numObjs        
        self.totalTrials = len(self.trial_data)
        self.corr_type = 'pearson'
        
    def init_from_pickle(self, pkFile):
        f = open(pkFile, 'rb')
        data = pk.load(f)
        f.close()
        self.subj_data = data
        self.trial_data = self.preprocess(self.subj_data)
        self.totalTrials = len(self.trial_data)
        
    def setPopCM(self):
        if self.numResp == 2:
            self.popCM, self.CM_order = self.getPopCM2x2fast(self.trial_data)
        else:
            self.popCM, self.CM_order = self.getPopCM(self.trial_data)
    
    def preprocess(self, subj_data, obj, trialNum): 
    # before the fb experiment, the HvM metadata, uploaded urls dont have unique hash id in the url, after feedback exp, both meta and the pushed json files changed
        RV = [] #Response vector
        SV = [] #Stimulus vector
        DV = [] #Distractor vector
        if obj=='face':
            RV = [] #Response vector
            DV = [] #Distractor vector
            RT = []
            for subj in self.subj_data: # subj is dict in list subj_data; to access string values in a dist within a list, use subj_data[0]['Response']
                models_name = np.unique(subj['Response'])
                models_size = np.unique(subj['Size'])
                self.models = []
                for idx1 in models_name:
                    for idx2 in models_size:
                        self.models.append([str(idx1)+'_'+str(idx2)])
                models_idxs = {}
                for idx, model in enumerate(self.models):
                    models_idxs[tuple(model)] = idx
                self.models_idxs = models_idxs

                for t_idx, t in enumerate(subj['RT']):
                    if t_idx>=trialNum[0] and t_idx<trialNum[1]:
                        RT.append(t)
                for r_idx, r in enumerate(subj['Response']):
                    if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                        RV.append([str(r)+'_'+str(subj['Size'][r_idx])])
                for s_idx, s in enumerate(subj['StimShown']):
                    if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                        DV.append([str(s)+'_'+str(subj['Size'][s_idx])])

        elif obj=='obj_lack':
            RV_s = [] #Response vector
            DV_s = [] #Distractor vector
            RV_p = [] 
            DV_p = []
            RV_r = [] 
            DV_r = []
            RV = []
            DV = []
            for subj in self.subj_data: # subj is dict in list subj_data; to access string values in a dist within a list, use subj_data[0]['Response']
                self.models = np.unique(subj['Response'])
                models_idxs = {}
                for idx, model in enumerate(self.models):
                    models_idxs[tuple(model)] = idx
                self.models_idxs = models_idxs

                for r_idx, r in enumerate(subj['Response']):
                    if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                        if subj['ImgData'][r_idx]['tname'] == 'obj_size':
                            RV_s.append(r)
                        elif subj['ImgData'][r_idx]['tname'] == 'position':
                            RV_p.append(r)
                        elif subj['ImgData'][r_idx]['tname'] == 'rotation':
                            RV_r.append(r)
                        else: #'objectome32'
                            RV.append(r)

                for s_idx, s in enumerate(subj['StimPresent']):
                    if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                        if subj['ImgData'][s_idx]['tname'] == 'obj_size':
                            DV_s.append(s)
                        elif subj['ImgData'][s_idx]['tname'] == 'position':
                            DV_p.append(s)
                        elif subj['ImgData'][s_idx]['tname'] == 'rotation':
                            DV_r.append(s)
                        else:
                            DV.append(s)

        elif obj=='obj':
            RV_s = [] #Response vector
            DV_s = [] #Distractor vector
            RV_p = [] 
            DV_p = []
            RV_r = [] 
            DV_r = []
            RV = []
            DV = []
            for subj in self.subj_data: # subj is dict in list subj_data; to access string values in a dist within a list, use subj_data[0]['Response']
                self.models = np.unique(subj['Response'])
                models_idxs = {}
                for idx, model in enumerate(self.models):
                    models_idxs[tuple(model)] = idx
                self.models_idxs = models_idxs

                for r_idx, r in enumerate(subj['Response']):
                    if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                        if subj['ImgData'][r_idx][0]['tname'] == 'obj_size':
                            RV_s.append(r)
                        elif subj['ImgData'][r_idx][0]['tname'] == 'position':
                            RV_p.append(r)
                        elif subj['ImgData'][r_idx][0]['tname'] == 'rotation':
                            RV_r.append(r)
                        else: #'objectome32'
                            RV.append(r)

                for s_idx, s in enumerate(subj['StimPresent']):
                    if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                        if subj['ImgData'][s_idx][0]['tname'] == 'obj_size':
                            DV_s.append(s)
                        elif subj['ImgData'][s_idx][0]['tname'] == 'position':
                            DV_p.append(s)
                        elif subj['ImgData'][s_idx][0]['tname'] == 'rotation':
                            DV_r.append(s)
                        else:
                            DV.append(s)

        elif obj=='2way':

            RV = [] #Response vector
            DV = [] #Distractor vector
            RV_s = [] #Response vector
            DV_s = [] #Distractor vector
            SV_s = []
            SV = []
            for subj in self.subj_data:

                for t_idx, t in enumerate(subj['ImgData']):
                    if t_idx>=trialNum[0] and t_idx<trialNum[1]:
                        if subj['ImgData'][t_idx][0]['tname'] == 'obj_size':
                            SV_s.append([t[1]['obj'],t[2]['obj']])
                        else: #'objectome32'
                            SV.append([t[1]['obj'],t[2]['obj']])

                for r_idx, r in enumerate(subj['Response']):
                    if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                        if subj['ImgData'][r_idx][0]['tname'] == 'obj_size':
                            RV_s.append(r)
                        else: #'objectome32'
                            RV.append(r)

                for s_idx, s in enumerate(subj['StimPresent']):
                    if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                        if subj['ImgData'][s_idx][0]['tname'] == 'obj_size':
                            DV_s.append(s)
                        else:
                            DV.append(s)

        elif obj=='2way_face':

            RV = [] #Response vector
            DV = [] #Distractor vector
            RV_s = [] #Response vector
            DV_s = [] #Distractor vector
            SV_s = []
            SV = []
            for subj in self.subj_data:

                for t_idx, t in enumerate(subj['ImgData']):
                    if t_idx>=trialNum[0] and t_idx<trialNum[1]:
                        if subj['ImgData'][t_idx][0]['var'] == 'V0_size':
                            SV_s.append([t[1]['obj'],t[2]['obj']])
                        else: #'objectome32'
                            SV.append([t[1]['obj'],t[2]['obj']])

                for r_idx, r in enumerate(subj['Response']):
                    if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                        if subj['ImgData'][r_idx][0]['var'] == 'V0_size':
                            RV_s.append(r)
                        else: #'objectome32'
                            RV.append(r)

                for s_idx, s in enumerate(subj['StimPresent']):
                    if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                        if subj['ImgData'][s_idx][0]['var'] == 'V0_size':
                            DV_s.append(s)
                        else:
                            DV.append(s)


        else:
            RV = [] #Response vector
            DV = [] #Distractor vector
            for subj in subj_data: # subj is dict in list subj_data; to access string values in a dist within a list, use subj_data[0]['Response']
                self.models = np.unique(subj['TestStim'])
                models_idxs = {}
                for idx, model in enumerate(self.models):
                    models_idxs[tuple(model)] = idx
                self.models_idxs = models_idxs

                for r_idx, r in enumerate(subj['Response']):
                    if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                        RV.append(r)
                for s_idx, s in enumerate(subj['StimPresent']):
                    if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                        DV.append(s)


        if obj=='obj':
            new_data_s = []
            new_data_p = []
            new_data_r = []
            new_data = []
            for idx, shown in enumerate(DV_s):
                model = shown
                CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
                CF_row_idx = self.models_idxs[tuple(RV_s[idx])] #response
                new_data_s.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
            for idx, shown in enumerate(DV_p):
                model = shown
                CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
                CF_row_idx = self.models_idxs[tuple(RV_p[idx])] #response
                new_data_p.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
            for idx, shown in enumerate(DV_r):
                model = shown
                CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
                CF_row_idx = self.models_idxs[tuple(RV_r[idx])] #response
                new_data_r.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
            for idx, shown in enumerate(DV):
                model = shown
                CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
                CF_row_idx = self.models_idxs[tuple(RV[idx])] #response
                new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
            return [new_data_s, new_data_p, new_data_r, new_data]

        elif obj=='2way':
            new_data_s = []
            new_data = []
            for idx, shown in enumerate(DV_s):
                model = shown
                CF_col_idx = self.models_idxs[model] #stimulus shown
                CF_row_idx = self.models_idxs[RV_s[idx]] #response
                new_data_s.append([CF_col_idx, CF_row_idx, [self.models_idxs[m] for m in SV_s[idx]]]) #order is shown, picked, distractors
            for idx, shown in enumerate(DV):
                model = shown
                CF_col_idx = self.models_idxs[model] #stimulus shown
                CF_row_idx = self.models_idxs[RV[idx]] #response
                new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[m] for m in SV[idx]]]) #order is shown, picked, distractors
            return [new_data_s, new_data]

        elif obj=='2way_face':
            new_data_s = []
            new_data = []
            for idx, shown in enumerate(DV_s):
                model = shown
                CF_col_idx = self.models_idxs[model] #stimulus shown
                CF_row_idx = self.models_idxs[RV_s[idx]] #response
                new_data_s.append([CF_col_idx, CF_row_idx, [self.models_idxs[m] for m in SV_s[idx]]]) #order is shown, picked, distractors
            for idx, shown in enumerate(DV):
                model = shown
                CF_col_idx = self.models_idxs[model] #stimulus shown
                CF_row_idx = self.models_idxs[RV[idx]] #response
                new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[m] for m in SV[idx]]]) #order is shown, picked, distractors
            return [new_data_s, new_data]

        elif obj=='face':
            new_data = []
            for idx, shown in enumerate(DV):
                if RT[idx]<3000:
                    model = shown
                    CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
                    CF_row_idx = self.models_idxs[tuple(RV[idx])] #response
                    new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
            return new_data

        else:
            new_data = []
            for idx, shown in enumerate(DV):
                model = shown
                CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
                CF_row_idx = self.models_idxs[tuple(RV[idx])] #response
                new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
            return new_data


    
    def getPopCM2x2fast(self, trial_data):
        combs = list(itertools.combinations(range(0, self.numObjs), 2))
        CMs = {}
        for c in combs:
            CMs[c] = np.zeros((2,2))
        for t in trial_data: # each trial can only increase +1 in total; statistics is based on many trials
            target = t[0]
            pick = t[1]
            cm = tuple(sorted(t[2])) #Because itertools always spits out the combs in sorted order; the two-way task is designed for each pair, either target is presented with equal times
            if target == cm[0]: #stimulus = True: when the signal present
                if target == pick: #response = true; Hit
                    CMs[cm][0,0] += 1
                else: # response = False; Miss
                    CMs[cm][1,0] += 1
            else: # stimulus = False; when the signal does not present
                if target == pick: # response = false; correct rejection
                    CMs[cm][1,1] += 1 
                else: # response = true; false alarm
                    CMs[cm][0,1] += 1
        return [CMs[c] for c in combs], combs
    
    def getPopCM(self, trial_data, order=[]): # trial_data is for individual subj or for all subj (myresult.trial_data)
        if len(trial_data[0][2]) != len(self.trial_data[0][2]):
            numResp = len(trial_data[0][2]) # should not use self.trial_data
        else:
            numResp = len(self.trial_data[0][2])
        #print numResp
        obj_inds = []
        for t in trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])

        if len(np.unique(obj_inds)) != self.numObjs:
            obj_inds = range(self.numObjs)
        else:
            obj_inds = obj_inds

        combs = list(itertools.combinations(np.unique(obj_inds), numResp))  
        CMs = [np.zeros((numResp, numResp)) for i in range(0, len(combs))]
        for trial in trial_data:
            distractor = [m for m in trial[2] if m != trial[0]]
            target = trial[0]
            pick = trial[1]
            possCombs = [[comb, idx] for idx, comb in enumerate(combs) if target in comb]
            for comb in possCombs:
                if set(distractor).issubset(set(comb[0])):
                    if len(order) > 0:
                        comb[0] = order
                    if pick == target:
                        idx = comb[0].index(pick)
                        CMs[comb[1]][idx, idx] += 1
                    elif pick != target:
                        CMs[comb[1]][comb[0].index(pick), comb[0].index(target)] += 1
                    else:
                        print('Matrix Error')
        return CMs, combs


    def getexposureCM(self, trial_data, trialNum, expoNum): # trial_data is for individual subj or for all subj (myresult.trial_data)
        if len(trial_data[0][2]) != len(self.trial_data[0][2]):
            numResp = len(trial_data[0][2]) # should not use self.trial_data
        else:
            numResp = len(self.trial_data[0][2])
        #print numResp
        obj_inds = []
        for t in trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])

        condi = self.subj_data[0]['Combinations']
        newcondi = []
        s1 = set(['NONSWAP', 'SWAP'])
        for subj in self.subj_data:
            s2 = set(subj.keys())
            for s in subj[list(s1.intersection(s2))[0]]:
                newcondi.append([x for idx, x in enumerate(condi[int(s)]) if idx>= expoNum[0] and idx<expoNum[1]]) #need to modify if the total number of condtion change

        if len(newcondi) != len(trial_data):
            print('trial number inconsistent')
        else:
            print(str(len(trial_data)))

        RV = [] #Response vector
        DV = [] #Distractor vector

        for subj in self.subj_data: # subj is dict in list subj_data; to access string values in a dist within a list, use subj_data[0]['Response']
            models = np.unique(subj['Response'])
            self.models = []
            for idx in models:
                self.models.append(idx)
            models_idxs = {}
            for idx, model in enumerate(self.models):
                models_idxs[tuple(model)] = idx
            self.models_idxs = models_idxs

            for r_idx, r in enumerate(subj['Response']):
                if r_idx>=trialNum[0] and r_idx<trialNum[1]:
                    RV.append(r)
            for s_idx, s in enumerate(subj['StimShown']):
                if s_idx>=trialNum[0] and s_idx<trialNum[1]:
                    DV.append(s)

        new_data = []
        for idx, shown in enumerate(DV):
            model = shown
            CF_col_idx = self.models_idxs[tuple(model)] #stimulus shown
            CF_row_idx = self.models_idxs[tuple(RV[idx])] #response
            new_data.append([CF_col_idx, CF_row_idx, [self.models_idxs[tuple(m)] for m in self.models]]) #order is shown, picked, distractors
        return newcondi, new_data
               
    def computeSplitHalf_size(self, numSplits, subsample, verbose = False, correct = True, plot_ = False): #subsample equal to total trial number if don't want to subsample
        import scipy.stats
        trial_data = self.trial_data
        Rs = []
        for s in range(0, numSplits):
            if verbose == True:
                print(s)
            else:
                pass
            np.random.shuffle(trial_data)
            if int(subsample)%2 == 0:
                half1.extend(t[0:subsample/2])
                half2.extend(t[-subsample/2:])
            else:
                half1.extend(t[0:subsample/2+1])
                half2.extend(t[-subsample/2:])

            if self.numResp == 2:
                CM1, combs = self.getPopCM2x2fast(half1)
                CM2, combs = self.getPopCM2x2fast(half2)
            else:
                CM1, combs = self.getPopCM(half1)
                CM2, combs = self.getPopCM(half2)
            half1_array = []
            half2_array = []
            for mat in range(0, len(CM1)):
                newarray = np.reshape(normalize_CM(CM1[mat]),(CM1[mat].shape[0]*CM1[mat].shape[1],-1))
                half1_array += list([x for x in newarray if x!=0])
                newarray = np.reshape(normalize_CM(CM2[mat]),(CM2[mat].shape[0]*CM2[mat].shape[1],-1))
                half2_array += list([x for x in newarray if x!=0])
            if self.corr_type == 'pearson':
                Rs.append(scipy.stats.pearsonr(half1_array, half2_array)[0])
                #correct = False
            else:
                Rs.append(scipy.stats.spearmanr(half1_array, half2_array)[0])
            if plot_ == True:
                plt.plot(half1_array, half2_array, 'b.')
        if correct == False:
            return Rs
        else:
            Rs_c = [SBcorrection(r, 2) for r in Rs]
            return Rs_c

    def computeSplitHalf_dprime(self, pair_trial_data, boot, starttrial, verbose = False, correct = True, plot_ = False, trial_data = None): #subsample equal to total trial number if don't want to subsample
        import scipy.stats

        count = [len(trial) for trial in pair_trial_data]

        corr_dprime = []
        for i in range(boot):
            temp = []
            for w in range(min(count)-starttrial+1):
                a = [random.sample(trial, w+starttrial) for trial in pair_trial_data]
                subsample = len(a[0])
                Rs = []
                for b in range(boot):
                    half1 = []
                    half2 = []
                    for t in a:
                        np.random.shuffle(t)
                    if int(subsample)%2 == 0:
                        half1.extend(t[0:subsample/2])
                        half2.extend(t[-subsample/2:])
                    else:
                        half1.extend(t[0:subsample/2+1])
                        half2.extend(t[-subsample/2:])
                    CM1, combs = self.getPopCM2x2fast(half1)
                    CM2, combs = self.getPopCM2x2fast(half2)
                    
                    half1_dprime = []
                    half2_dprime = []
                    for mat in range(0, len(CM1)):
                        half1_dprime.append(d_prime2x2(CM1[mat])) # previously normalized CM, which caused nan when divided by 0
                        half2_dprime.append(d_prime2x2(CM2[mat]))
                        
                    Rs.append(scipy.stats.spearmanr(half1_dprime, half2_dprime)[0])
                    
                temp.append(np.ma.masked_invalid(Rs).mean(0))
            corr_dprime.append(temp)
            return corr_dprime

    def computeSplitHalf(self, numSplits, subsample, verbose = False, correct = True, plot_ = False, trial_data = None): #subsample equal to total trial number if don't want to subsample
        import scipy.stats
        if trial_data == None:
            trial_data = self.trial_data
        else:
            trial_data = trial_data

        Rs = []
        for s in range(0, numSplits):
            if verbose == True:
                print(s)
            else:
                pass
            np.random.shuffle(trial_data)

            half1 = []
            half2 = []
            if int(subsample)%2 == 0:
                half1.extend(trial_data[0:subsample/2])
                half2.extend(trial_data[-subsample/2:])
            else:
                half1.extend(trial_data[0:subsample/2+1])
                half2.extend(trial_data[-subsample/2:])

            if self.numResp == 2:
                CM1, combs = self.getPopCM2x2fast(half1)
                CM2, combs = self.getPopCM2x2fast(half2)
            else:
                CM1, combs = self.getPopCM(half1)
                CM2, combs = self.getPopCM(half2)

            half1_array = []
            half2_array = []
            for mat in range(0, len(CM1)):
                half1_array += list(normalize_CM(CM1[mat])[np.eye(CM1[mat].shape[0])==0])
                half2_array += list(normalize_CM(CM2[mat])[np.eye(CM2[mat].shape[0])==0])
            if self.corr_type == 'pearson':
                Rs.append(scipy.stats.pearsonr(half1_array, half2_array)[0])
                #correct = False
            else:
                Rs.append(scipy.stats.spearmanr(half1_array, half2_array)[0])
            if plot_ == True:
                plt.plot(half1_array, half2_array, 'b.')
        if correct == False:
            return Rs
        else:
            Rs_c = [SBcorrection(r, 2) for r in Rs]
            return Rs_c
      
    def imputeNtoM(self, use_objects):
        #Produces a single imputed matrix of a given size for given objects. The matrix will have blank entries
        #if you ask for a greater size than is given by the number of objects represented by your data
        obj_inds = []
        for t in self.trial_data:
            if len(np.unique(obj_inds)) == self.numObjs:
                break
            else:
                obj_inds.append(t[0])
        t = []
        for obj in use_objects:
            t.append(self.models.index(obj))
        import itertools
        combs = list(itertools.combinations(t, self.numResp))
        CM_imputed = np.zeros((len(t),len(t)))
        for trial in self.trial_data:
            for comb in combs:
                if set(comb).issubset(set(trial[2])):
                    if trial[0] == trial[1]:
                        CM_imputed[t.index(trial[0]), t.index(trial[0])] += 1
                    else:
                        CM_imputed[t.index(trial[1]), t.index(trial[0])] += 1
        return CM_imputed


    