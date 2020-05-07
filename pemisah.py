# -*- coding: utf-8 -*-
"""
Created on Sun May  5 01:43:10 2019

@author: asus
"""
#import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import scipy.signal as sp

data = pd.read_pickle('Feature Data.pk')
#data = data.sort_values(['Diagnosis'],axis=0,ascending=True)
data = data.sample(frac=1)
train_labels = data.loc[:,'Diagnosis'].values
train_labels = train_labels.astype(int)


sort_data = data.sort_values(['Diagnosis'],axis=0,ascending=True)

sort_data_a = sort_data.drop(columns = ['Diagnosis','GLCM_mat','Normalize_Hist'])
sort_data_a = sort_data_a.values.astype(float)

col_bin = 16
sort_data_b = sort_data.loc[:,'Normalize_Hist'].values
sort_data_b = np.concatenate(sort_data_b)
sort_data_b = np.reshape(sort_data_b,(-1,col_bin,col_bin,col_bin))

empty_b = np.zeros(np.shape(sort_data_b))
for n in range (np.shape(sort_data_b)[0]):
     empty_b[n,:,:,:] = sort_data_b[n,:,:,:]/np.sum(sort_data_b[n,:,:,:])
sort_data_b = empty_b
sort_data_b = sort_data_b/np.max(sort_data_b)

sort_data_a = np.nan_to_num(sort_data_a)
sort_data_b = np.nan_to_num(sort_data_b)

for i in range (np.shape(sort_data_a)[1]):
     sort_data_a[:,i] = sort_data_a[:,i]/np.max(sort_data_a[:,i])

sort_train_labels = sort_data.loc[:,'Diagnosis'].values
sort_train_labels = sort_train_labels.astype(int)
sort_y_binary = keras.utils.to_categorical(sort_train_labels).astype(int)

test_data_n = 200

train_data_a = np.concatenate([sort_data_a[test_data_n:-test_data_n,:]])
train_data_b = np.concatenate([sort_data_b[test_data_n:-test_data_n,:,:,:]])
train_y = np.concatenate([sort_y_binary[test_data_n:-test_data_n,:]])

test_data_a = np.concatenate([sort_data_a[0:test_data_n,:],sort_data_a[-test_data_n:,:]])
test_data_b = np.concatenate([sort_data_b[0:test_data_n,:,:,:],sort_data_b[-test_data_n:,:,:,:]])
test_y = np.concatenate([sort_y_binary[0:test_data_n,:],sort_y_binary[-test_data_n:,:]])



np.save('sort_data_a', sort_data_a)
np.save('sort_data_b', sort_data_b)
np.save('sort_y_binary', sort_y_binary)


np.save('test_data_a', test_data_a)
np.save('test_data_b', test_data_b)
np.save('test_y', test_y)

np.save('train_data_a', train_data_a)
np.save('train_data_b', train_data_b)
np.save('train_y', train_y)


