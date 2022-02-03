import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from sklearn.model_selection import train_test_split
import pickle
import random
from preprocessing import *
from tensorflow import keras

def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def normalize(img):
  '''
  Normalizes an array
  (subtract mean and divide by standard deviation)
  '''
  eps = 0.001
  # print(np.shape(img))
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

path = './data/physionet/validation/'
files = glob.glob(path + '/*.wav')
os.chdir('D:\\UW\\Final thesis\\audio_classification')
filename= './model/model/pascal_40_5_1_200_128_01_31_2022_22_56_03'
model = keras.models.load_model(filename)
preprocess= preprocessing()
# for file in files:
#
#     with open('answers.txt', 'a') as of:
#         data = preprocess.extract_features(file)
#         data = np.array(data)
#         data = np.nan_to_num(data)
#         data = normalize_dataset(data)
#         print("shape xdata:", np.shape(data))
#         data =np.reshape(data,[1,np.shape(data)[0],np.shape(data)[1]])
#         print("shape xdata:", np.shape(data))
#         q, _ = model.predict(data, verbose=0)
#         y_pred = np.argmax(q, axis=1)
#         print(y_pred)
#         head, tail = os.path.split(file)
#         tail=tail.split('.')[0]
#         if y_pred[0]==1:
#
#             of.write(tail+',1\n')
#         else:
#             of.write(tail+',-1\n')
#         print(file)
