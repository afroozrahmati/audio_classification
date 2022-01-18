from preprocessing import *
import flwr as fl
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,mean_squared_error,mutual_info_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
from sklearn.metrics.cluster import normalized_mutual_info_score
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
from clients_data_generation import *

def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def load_processed_data(idx,total_no_clients):

    pathnormal= './data/physionet/normal/'
    pathabnormal = './data/physionet/abnormal/'
    p = preprocessing()
    #last client index is for server evaluation data
    x_train, x_test, y_train, y_test, start,end = p.load_processed_partition(idx, total_no_clients)
    #p.load_data(pathnormal, pathabnormal, total_no_clients, total_no_clients)

    print("train shape: ", np.shape(x_train))
    print("test shape: ", np.shape(x_test))
    print("train label shape: ",np.shape(y_train))
    print("test label shape: ",np.shape(y_test) )

    x_train = np.asarray(x_train)
    x_train = np.nan_to_num(x_train)
    x_test = np.asarray(x_test)
    x_test = np.nan_to_num(x_test)

    y_train = np.asarray(y_train)
    y_train = np.nan_to_num(y_train)

    return x_train, x_test, y_train, y_test


p = preprocessing()
# p.rename_files('.\\data\\physionet\\normal')
# p.rename_files('.\\data\\physionet\\abnormal')
x_train, x_test, y_train, y_test, start, end = p.load_processed_train_data( 10)
print("i=", 10)
print("start=", start)
print("end=", end)
print("train shape: ", np.shape(x_train))
print("test shape: ", np.shape(x_test))
print("train label shape: ", np.shape(y_train))
print("test label shape: ", np.shape(y_test))

print("--------------")

client_counts = 10
for i in range(0,client_counts):
    x_train, x_test, y_train, y_test, start, end = p.load_processed_partition(i, 10)
    print("i=",i)
    print("start=",start)
    print("end=",end)
    print("train shape: ", np.shape(x_train))
    print("test shape: ", np.shape(x_test))
    print("train label shape: ", np.shape(y_train))
    print("test label shape: ", np.shape(y_test))

    print ("--------------")



