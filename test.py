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

def load_processed_data(total_no_clients):

    pathnormal= './data/physionet/normal/'
    pathabnormal = './data/physionet/abnormal/'
    p = preprocessing()
    #last client index is for server evaluation data
    x_train, x_test, y_train, y_test = p.load_processed_partition(total_no_clients, total_no_clients)
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

def get_model(timesteps , n_features ):
    gamma = 1
    # tf.keras.backend.clear_session()
    print('Setting Up Model for training')
    print(gamma)

    inputs = Input(shape=(timesteps, n_features))
    encoder = LSTM(32, activation='tanh')(inputs)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(100, activation='relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder_out = Dense(100, activation=None, name='encoder_out')(encoder)
    clustering = ClusteringLayer(n_clusters=2, name='clustering', alpha=0.05)(encoder_out)
    hidden = RepeatVector(timesteps, name='Hidden')(encoder_out)
    decoder = Dense(100, activation='relu')(hidden)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = LSTM(32, activation='tanh', return_sequences=True)(decoder)
    output = TimeDistributed(Dense(n_features), name='decoder_out')(decoder)
    encoder_model = Model(inputs=inputs, outputs=encoder_out)
    # kmeans.fit(encoder_model.predict(x_train))

    model = Model(inputs=inputs, outputs=[clustering, output])

    clustering_model = Model(inputs=inputs, outputs=clustering)

    # plot_model(model, show_shapes=True)
    #model.summary()
    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    model.compile(loss={'clustering': 'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer='adam',
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})

    print('Model compiled.           ')
    return model


p = preprocessing()
# p.rename_files('.\\data\\physionet\\normal')
# p.rename_files('.\\data\\physionet\\abnormal')
x_train, x_test, y_train, y_test  = p.load_processed_partition(4, 10) #load_processed_data(0,1)

print("train shape: ", np.shape(x_train))
print("test shape: ", np.shape(x_test))
print("train label shape: ", np.shape(y_train))
print("test label shape: ", np.shape(y_test))

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

timesteps = np.shape(x_train)[1]
n_features = np.shape(x_train)[2]
print("timesteps:", timesteps)
print("n_features:", n_features)
model = get_model(timesteps, n_features)


batch_size: int = 64
epochs: int = 400

        # Train the model using hyperparameters from config
# history = model.fit(x_train,
#         y={'clustering': y_train, 'decoder_out': x_train},
#         epochs=epochs,
#         validation_split=0.2,
#         # validation_data=(x_test, (y_test, x_test)),
#         batch_size=batch_size,
#         )

train_history = model.fit(x_train,
                          y={'clustering': y_train, 'decoder_out': x_train},
                          epochs=epochs,
                          validation_split=0.2,
                          # validation_data=(x_test, (y_test, x_test)),
                          batch_size=batch_size,
                          verbose=2
                         )

q, _ = model.predict(x_train, verbose=0)
q_t, _ = model.predict(x_test, verbose=0)
p = target_distribution(q)

y_pred = np.argmax(q, axis=1)
y_arg = np.argmax(y_train, axis=1)
y_pred_test = np.argmax(q_t, axis=1)
y_arg_test = np.argmax(y_test, axis=1)
# acc = np.sum(y_pred == y_arg).astype(np.float32) / y_pred.shape[0]
# testAcc = np.sum(y_pred_test == y_arg_test).astype(np.float32) / y_pred_test.shape[0]
# accuracy = np.round(accuracy_score(y_arg, y_pred), 5)
accuracy = np.round(accuracy_score(y_arg_test, y_pred_test), 5)
kld_loss = np.round(mutual_info_score(y_arg_test, y_pred_test), 5)


print('====================')
print('====================')
print('====================')
print('====================')
print('Train accuracy')
print(accuracy)
