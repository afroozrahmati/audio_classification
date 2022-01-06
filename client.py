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

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(self.x_train,
                y={'clustering': self.y_train, 'decoder_out': self.x_train},
                epochs=epochs,
                validation_split=0.1,
                # validation_data=(x_test, (y_test, x_test)),
                batch_size=batch_size,
                )


        # history = self.model.fit(
        #     self.x_train,
        #     self.y_train,
        #     batch_size,
        #     epochs,
        #     validation_split=0.1,
        # )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["clustering_accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_clustering_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        #loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)

        q_t, _ = self.model.predict(self.x_test, verbose=0)
        y_pred_test = np.argmax(q_t, axis=1)
        y_arg_test = np.argmax(self.y_test, axis=1)
        accuracy = np.round(accuracy_score(y_arg_test, y_pred_test), 5)
        # mse_loss = np.round(mean_squared_error(y_arg_test, y_pred_test), 5)
        kld_loss = np.round(mutual_info_score(y_arg_test, y_pred_test), 5)

        num_examples_test = len(self.x_test)
        return kld_loss, num_examples_test, {"accuracy": accuracy}


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

def main() -> None:
    # Parse command line argument `partition`
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    # args = parser.parse_args()
    #file_path_normal =  sys.argv[1] #    #+ sys.argv[0]
    #file_path_abnormal = sys.argv[2] #  #+ sys.argv[1]
    # Load and compile Keras model

    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    # args = parser.parse_args()

    idx = int(sys.argv[1])
    clients_count=int(sys.argv[2])
    x_train, x_test, y_train, y_test  = load_processed_data(idx,clients_count)
    x_train = np.asarray(x_train)
    timesteps = np.shape(x_train)[1]
    n_features = np.shape(x_train)[2]
    print("timesteps:",timesteps)
    print("n_features:", n_features)
    model= get_model(timesteps,n_features)


    #clients_count = 10 #int(sys.argv[1]) #10 #sys.argv[2]


    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("192.168.1.237:8080", client=client)


def load_processed_data(clinet_index,total_no_clients):

    pathnormal= './data/physionet/normal/'
    pathabnormal = './data/physionet/abnormal/'
    p = preprocessing()
    #last client index is for server evaluation data
    x_train, x_test, y_train, y_test = p.load_processed_partition(clinet_index, total_no_clients)
    #p.load_data(pathnormal, pathabnormal, clinet_index, total_no_clients)

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




if __name__ == "__main__":
    main()
