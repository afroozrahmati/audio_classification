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

with open('physionet_MFCC40t_X.pkl', 'rb') as input:
    data_x = pickle.load(input)

with open('physionet_MFCC40t_Y.pkl', 'rb') as input:
    data_y = pickle.load(input)

data_x = np.array(data_x)
data_x = np.nan_to_num(data_x)
data_x = normalize_dataset(data_x)
data_y = np.asarray(data_y)
print(np.shape(data_x))

print(np.shape(data_y))

optimizer = Adam(0.0001, beta_1=0.1, beta_2=0.001, amsgrad=True)
#optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

n_classes = 2
batch_size = 64
epochs = 200
#gamma =4
callbacks = EarlyStopping(monitor='val_clustering_accuracy', mode='max',
                              verbose=2, patience=800, restore_best_weights=True)

model_dir = './model/'
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
timesteps = np.shape(x_train)[1]
n_features = np.shape(x_train)[2]
print((timesteps, n_features))

x_train = np.asarray(x_train)
x_test = np.nan_to_num(x_test)
x_test = np.asarray(x_test)

for gamma in  [0,0.5,1,3,5,7,9,10]:
    os.chdir('D:\\UW\\Final thesis\\audio_classification')
    now = datetime.now() # current date and time
    now =now.strftime("%m")+'_'+now.strftime("%d")+'_'+now.strftime("%Y")+'_'+now.strftime("%H")+'_'+now.strftime("%M")+'_'+now.strftime("%S")

    tf.keras.backend.clear_session()
    print('Setting Up Model for training')
    print(gamma)
    model_name = now +'_'+ 'Gamma('+str(gamma) +')-Optim('+"Adam"+')'+'_'+str(epochs)
    print(model_name)

    model = 0

    inputs=encoder=decoder=hidden=clustering=output=0

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

    # kmeans = KMeans(n_clusters=2, n_init=100)

    encoder_model = Model(inputs=inputs, outputs=encoder_out)
    # kmeans.fit(encoder_model.predict(x_train))


    model = Model(inputs=inputs, outputs=[clustering, output])


    clustering_model = Model(inputs=inputs, outputs=clustering)

    #plot_model(model, show_shapes=True)
    model.summary()
    q, _ = model.predict(x_train, verbose=2)
    q_t, _ = model.predict(x_test, verbose=2)
    p = target_distribution(q)

    y_pred = np.argmax(p, axis=1)
    y_arg = np.argmax(y_train, axis=1)
    acc = np.round(accuracy_score(y_arg, y_pred), 5)

    print('====================')
    print('====================')
    print('====================')
    print('====================')
    print('Pre Training Accuracy')
    print(acc)
    print('====================')
    print('====================')
    print('====================')
    print('====================')

    model.compile(loss={'clustering': 'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer='adam',
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})
    print('Model compiled.')
    print('Training Starting:')
    train_history = model.fit(x_train,
                              y={'clustering': y_train, 'decoder_out': x_train},
                              epochs=epochs,
                              validation_split=0.2,
                              # validation_data=(x_test, (y_test, x_test)),
                              batch_size=batch_size,
                              verbose=2,
                              callbacks=callbacks)


    q, _ = model.predict(x_train, verbose=0)
    q_t, _ = model.predict(x_test, verbose=0)
    p = target_distribution(q)

    y_pred = np.argmax(q, axis=1)
    y_arg = np.argmax(y_train, axis=1)
    y_pred_test = np.argmax(q_t, axis=1)
    y_arg_test = np.argmax(y_test, axis=1)
    # acc = np.sum(y_pred == y_arg).astype(np.float32) / y_pred.shape[0]
    # testAcc = np.sum(y_pred_test == y_arg_test).astype(np.float32) / y_pred_test.shape[0]
    acc = np.round(accuracy_score(y_arg, y_pred), 5)
    testAcc = np.round(accuracy_score(y_arg_test, y_pred_test), 5)

    nmi = np.round(normalized_mutual_info_score(y_arg, y_pred), 5)
    nmi_test = np.round(normalized_mutual_info_score(y_arg_test, y_pred_test), 5)
    ari = np.round(adjusted_rand_score(y_arg, y_pred), 5)
    ari_test = np.round(adjusted_rand_score(y_arg_test, y_pred_test), 5)
    print('====================')
    print('====================')
    print('====================')
    print('====================')
    print('Train accuracy')
    print(acc)
    print('Test accuracy')
    print(testAcc)

    print('NMI')
    print(nmi)
    print('ARI')
    print(ari)
    print('====================')
    print('====================')
    print('====================')
    print('====================')

    result = "Gamma="+str(gamma)+', Epochs='+str(epochs)+ ', Lr='+'0.0001, '+'NMI='+str(nmi) +', ARI='+str(ari) +', Train accuracy='+str(acc) + ', Test accuracy='+ str(testAcc)+"\n"
    with open('result.txt', 'a') as f:
        f.write(result)

    os.chdir('D:\\UW\\Final thesis\\audio_classification')
    saved_format = {
        'history': train_history.history,
        'gamma': gamma,
        'lr': K.eval(model.optimizer.lr),
        'batch': batch_size,
        'accuracy': acc,
        'nmi': nmi,
        'ari': ari,
        'nmi_test': nmi_test,
        'ari_test': ari_test,
        'test_accuracy': testAcc,
    }

    os.chdir(model_dir)
    pklName = model_name + '.pkl'
    # saved_format = [train_history.history, gamma, K.eval(model.optimizer.lr), batch_size]
    # with open(pklName, 'wb') as out_file:
    #     pickle.dump(train_history.history, out_file, pickle.HIGHEST_PROTOCOL)
    with open(pklName, 'wb') as out_file:
        pickle.dump(saved_format, out_file, pickle.HIGHEST_PROTOCOL)

    print('Saving model.')
    save_name = './model/' + model_name
    model.save(save_name)

    os.chdir('D:\\UW\\Final thesis\\audio_classification')
    produce_plot(model_name, train_history.history, gamma, testAcc)