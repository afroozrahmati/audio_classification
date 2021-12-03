import os
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

# Make TensorFlow log less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# def get_model(timesteps , n_features ):
#     gamma = 6
#     # tf.keras.backend.clear_session()
#     print('Setting Up Model for training')
#     print(gamma)
#
#     inputs = Input(shape=(timesteps, n_features))
#     encoder = LSTM(32, activation='tanh')(inputs)
#     encoder = Dropout(0.2)(encoder)
#     encoder = Dense(64, activation='relu')(encoder)
#     encoder = Dropout(0.2)(encoder)
#     encoder = Dense(100, activation='relu')(encoder)
#     encoder = Dropout(0.2)(encoder)
#     encoder_out = Dense(100, activation=None, name='encoder_out')(encoder)
#     clustering = ClusteringLayer(n_clusters=2, name='clustering', alpha=0.05)(encoder_out)
#     hidden = RepeatVector(timesteps, name='Hidden')(encoder_out)
#     decoder = Dense(100, activation='relu')(hidden)
#     decoder = Dense(64, activation='relu')(decoder)
#     decoder = LSTM(32, activation='tanh', return_sequences=True)(decoder)
#     output = TimeDistributed(Dense(n_features), name='decoder_out')(decoder)
#     encoder_model = Model(inputs=inputs, outputs=encoder_out)
#     # kmeans.fit(encoder_model.predict(x_train))
#
#     model = Model(inputs=inputs, outputs=[clustering, output])
#
#     clustering_model = Model(inputs=inputs, outputs=clustering)
#
#     # plot_model(model, show_shapes=True)
#     model.summary()
#     optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
#     model.compile(loss={'clustering': 'kld', 'decoder_out': 'mse'},
#                   loss_weights=[gamma, 1], optimizer=optimizer,
#                   metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})
#
#     print('Model compiled.           ')
#     return model

# def load_processed_data(client_name):
#     data_process= clients_data_generation(client_name)
#     x_train,y_train,x_test,y_test = data_process.get_clients_data()
#
#     print("train shape: ", np.shape(x_train))
#     print("test shape: ", np.shape(x_test))
#     print("train label shape: ", y_train.shape)
#     print("test label shape: ", y_test.shape)
#
#     x_train = np.asarray(x_train)
#     x_test = np.nan_to_num(x_test)
#     x_test = np.asarray(x_test)
#     return x_train,y_train,x_test, y_test


# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
client_name ='client1' #sys.argv[1]  # #+ sys.argv[0]
print(client_name)
#x_train, y_train, x_test, y_test = load_processed_data(client_name)  # args.partition)

#model = get_model(128, 40)