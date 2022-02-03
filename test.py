# # from preprocessing import *
# # import flwr as fl
# # import tensorflow as tf
# # import argparse
# # import os
# # import sys
# # import numpy as np
# # import tensorflow as tf
# # from sklearn.metrics import accuracy_score
# # from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
# # from tensorflow.keras.optimizers import Adam
# # from sklearn.metrics import accuracy_score,mean_squared_error,mutual_info_score
# # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# # import librosa.display
# # import matplotlib.pyplot as plt
# # from scipy.io import wavfile
# # import pandas as pd
# # import glob
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy import signal
# # import tensorflow as tf
# # from keras import layers
# # from keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
# # from keras.models import Sequential, load_model
# # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # from tensorflow.keras.callbacks import Callback
# # from keras.models import Model
# # from tensorflow.keras.utils import plot_model
# # from tensorflow.keras.optimizers import Adam
# # from sklearn.metrics.cluster import normalized_mutual_info_score
# # from sklearn.metrics import accuracy_score
# # from sklearn.metrics.cluster import adjusted_rand_score
# # import keras.backend as K
# # from sklearn.model_selection import train_test_split
# # import pandas as pd
# # import numpy as np
# # import os
# # from tqdm import tqdm
# # from tqdm import tqdm_notebook
# # from datetime import datetime
# # import os, fnmatch
# # import pickle
# # from plots import produce_plot
# # from ClusteringLayer import *
# from clients_data_generation import *
#
# def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
#     weight = q ** 2 / q.sum(0)
#     return (weight.T / weight.sum(1)).T
#
# def load_processed_data(idx,total_no_clients):
#
#     pathnormal= './data/physionet/normal/'
#     pathabnormal = './data/physionet/abnormal/'
#     p = preprocessing()
#     #last client index is for server evaluation data
#     x_train, x_test, y_train, y_test, start,end = p.load_processed_partition(idx, total_no_clients)
#     #p.load_data(pathnormal, pathabnormal, total_no_clients, total_no_clients)
#
#     print("train shape: ", np.shape(x_train))
#     print("test shape: ", np.shape(x_test))
#     print("train label shape: ",np.shape(y_train))
#     print("test label shape: ",np.shape(y_test) )
#
#     x_train = np.asarray(x_train)
#     x_train = np.nan_to_num(x_train)
#     x_test = np.asarray(x_test)
#     x_test = np.nan_to_num(x_test)
#
#     y_train = np.asarray(y_train)
#     y_train = np.nan_to_num(y_train)
#
#     return x_train, x_test, y_train, y_test
#
# audio_path='./data/pascal/abnormal/murmur__244_1309198148498_B.wav'
# y, sr = librosa.load(audio_path, duration=5)
# print(sr)
# #
# # p = preprocessing()
# # # p.rename_files('.\\data\\physionet\\normal')
# # # p.rename_files('.\\data\\physionet\\abnormal')
# # x_train, x_test, y_train, y_test, start, end = p.load_processed_train_data( 10)
# # print("i=", 10)
# # print("start=", start)
# # print("end=", end)
# # print("train shape: ", np.shape(x_train))
# # print("test shape: ", np.shape(x_test))
# # print("train label shape: ", np.shape(y_train))
# # print("test label shape: ", np.shape(y_test))
# #
# # print("--------------")
# #
# # client_counts = 10
# # for i in range(0,client_counts):
# #     x_train, x_test, y_train, y_test, start, end = p.load_processed_partition(i, 10)
# #     print("i=",i)
# #     print("start=",start)
# #     print("end=",end)
# #     print("train shape: ", np.shape(x_train))
# #     print("test shape: ", np.shape(x_test))
# #     print("train label shape: ", np.shape(y_train))
# #     print("test label shape: ", np.shape(y_test))
# #
# #     print ("--------------")
# #
# #
# #

import keras
import pickle
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
from keras import layers
from keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
import keras.backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tqdm import tqdm_notebook
from datetime import datetime
import os, fnmatch
import pickle
from plots import produce_plot
from ClusteringLayer import *

def normalize(img):
    '''
    Normalizes an array
    (subtract mean and divide by standard deviation)
    '''
    eps = 0.001
    #print(np.shape(img))
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
    return img

def normalize_dataset(x):
    '''
    Normalizes list of arrays
    (subtract mean and divide by standard deviation)
    '''
    normalized_dataset = []
    for img in x:
        normalized = normalize(img)
        normalized_dataset.append(normalized)
    return normalized_dataset

def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

filename= r"D:\UW\Final thesis\audio_classification\model\model\physionet_40_5_1_200_128_02_01_2022_13_37_44"
model = keras.models.load_model(filename)
file= './data/processed_files/pascal_40_5_X.pkl'
file_y= './data/processed_files/pascal_40_5_Y.pkl'


with open(file, 'rb') as input:
    data_x = pickle.load(input)

with open(file_y, 'rb') as input:
    data_y = pickle.load(input)

data_x = np.array(data_x)
data_x = np.nan_to_num(data_x)
data_x = normalize_dataset(data_x)
data_y = np.asarray(data_y)
print(np.shape(data_x))

print(np.shape(data_y))

x_train, _,y_train,_  = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

print("shape x_train:", np.shape(x_train))


x_train = np.asarray(x_train)

#optimizer = Adam(0.0001, beta_1=0.1, beta_2=0.001, amsgrad=True)
#optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

n_classes = 2
batch_size = 64
epochs = 500

q, _ = model.predict(x_train, verbose=0)
#q_t, _ = model.predict(x_test, verbose=0)
p = target_distribution(q)
#p_t = target_distribution(q_t)
y_pred = np.argmax(p, axis=1)
y_arg = np.argmax(y_train, axis=1)
#y_pred_test = np.argmax(q_t, axis=1)
#y_arg_test = np.argmax(y_test, axis=1)
# acc = np.sum(y_pred == y_arg).astype(np.float32) / y_pred.shape[0]
# testAcc = np.sum(y_pred_test == y_arg_test).astype(np.float32) / y_pred_test.shape[0]
acc = np.round(accuracy_score(y_arg, y_pred), 5)
#testAcc = np.round(accuracy_score(y_arg_test, y_pred_test), 5)

nmi = np.round(normalized_mutual_info_score(y_arg, y_pred), 5)
#nmi_test = np.round(normalized_mutual_info_score(y_arg_test, y_pred_test), 5)
ari = np.round(adjusted_rand_score(y_arg, y_pred), 5)
#ari_test = np.round(adjusted_rand_score(y_arg_test, y_pred_test), 5)
ami = np.round(adjusted_mutual_info_score(y_arg, y_pred), 5)
#ami_test = np.round(adjusted_mutual_info_score(y_arg_test, y_pred_test), 5)
print('====================')
print('====================')
print('====================')
print('====================')
print('accuracy')
print(acc)

#print(testAcc)

print('NMI', nmi)

print('ARI', ari)

#print('NMI test', nmi_test)

#print('ARI test', ari_test)

print('AMI', ami)
#print('AMI test', ami_test)
print('====================')
print('====================')
print('====================')
print('====================')
