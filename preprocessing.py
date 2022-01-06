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

class preprocessing:
    def __init__(self,sr = 22050,duration = 5):
          # Sampling rate
        self.samples = sr * duration

    def normalize(self,img):
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

    def normalize_dataset(self,x):
      '''
      Normalizes list of arrays
      (subtract mean and divide by standard deviation)
      '''
      normalized_dataset = []
      for img in x:
          normalized = self.normalize(img)
          normalized_dataset.append(normalized)
      return normalized_dataset

    # 865
    def extract_features(self,audio_path):
        #     y, sr = librosa.load(audio_path, duration=3)
        y, sr = librosa.load(audio_path, duration=10)
        #     y = librosa.util.normalize(y)

        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)
        if len(y) > self.samples:  # long enough
            y = y[0:0 + self.samples]
        else:  # pad blank
            padding = self.samples - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, self.samples - len(y) - offset), 'constant')

        S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048,
                                           hop_length=865,
                                           n_mels=128)
        mfccs = np.transpose(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40))

        #     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return mfccs

    #rename files to the proper format, so we can partition it for each client
    def rename_files(self,path):
        files = glob.glob(path + '/*.wav')
        i = 0
        for file in files:
            new_name='.'+file.split('.')[1]+'-'+str(i)+'.wav'
            i+=1
            os.rename(file, new_name)

    #load related data for each client
    def load_data(self,pathnormal,pathabnormal,client_index, total_no_clients):

        # pathnormal= './data/physionet/normal/'
        # pathabnormal = './data/physionet/abnormal/'
        # we separate some part of our data for server evaluation, so we increase the No. of clients and keep last bunch for server evaluation purpose
        total_no_clients+=1
        x_data_normal = []
        y_data_normal =[]
        x_data_abnormal = []
        y_data_abnormal =[]

        files = glob.glob(pathnormal + '/*.wav')
        counts= len(files)//total_no_clients
        start = client_index  * counts
        end = ( client_index + 1 ) * counts
        for file in files:
            if start <= int(file.split('-')[1].split('.')[0]) < end:
                x_data_normal.append(self.extract_features(file))
                y_data_normal.append([1,0])
                print(file)


        files = glob.glob(pathabnormal + '/*.wav')
        counts= len(files)//total_no_clients
        start = client_index * counts
        end = ( client_index + 1 ) * counts
        for file in files:
            if start <= int(file.split('-')[1].split('.')[0]) < end:
                x_data_abnormal.append(self.extract_features(file))
                y_data_abnormal.append([0,1])
                print(file)

        # files = glob.glob(pathnormal + '/*.wav')
        # for file in files:
        #     x_data_normal.append(self.extract_features(file))
        #     y_data_normal.append([1,0])
        #     print(file)

        # files = glob.glob(pathabnormal + '/*.wav')
        # for file in files:
        #     x_data_abnormal.append(self.extract_features(file))
        #     y_data_abnormal.append([0,1])
        #     print(file)

        print("shape x normal data:", np.shape(x_data_normal) )
        print("shape y normal data:", np.shape(y_data_normal) )
        print("shape x abnormal data:", np.shape(x_data_abnormal)  )
        print("shape y abnormal data:", np.shape(y_data_abnormal)  )

        data_x = np.concatenate((x_data_normal,x_data_abnormal))
        data_y = np.concatenate((y_data_normal,y_data_abnormal))

        data_x = np.array(data_x)
        data_x = np.nan_to_num(data_x)
        data_x = self.normalize_dataset(data_x)
        data_y = np.asarray(data_y)

        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def process_files(self,pathnormal,pathabnormal):

        # pathnormal= './data/physionet/normal/'
        # pathabnormal = './data/physionet/abnormal/'
        # we separate some part of our data for server evaluation, so we increase the No. of clients and keep last bunch for server evaluation purpose

        x_data_normal = []
        y_data_normal =[]
        x_data_abnormal = []
        y_data_abnormal =[]

        files = glob.glob(pathnormal + '/*.wav')

        for file in files:
            x_data_normal.append(self.extract_features(file))
            y_data_normal.append([1,0])
            print(file)


        files = glob.glob(pathabnormal + '/*.wav')
        for file in files:
            x_data_abnormal.append(self.extract_features(file))
            y_data_abnormal.append([0,1])
            print(file)

        data_x = np.concatenate((x_data_normal, x_data_abnormal))
        data_y = np.concatenate((y_data_normal, y_data_abnormal))

        print("shape xdata:", np.shape(data_x))
        print("shape y data:", np.shape(data_y))

        with open('physionet_MFCC40t_X.pkl', 'wb') as o:
            pickle.dump(data_x, o, pickle.HIGHEST_PROTOCOL)

        with open('physionet_MFCC40t_Y.pkl', 'wb') as o:
            pickle.dump(data_y, o, pickle.HIGHEST_PROTOCOL)



    def load_processed_partition(self,client_index, total_no_clients):
        with open('physionet_MFCC40t_X.pkl', 'rb') as input:
            data_x = pickle.load(input)

        with open('physionet_MFCC40t_Y.pkl', 'rb') as input:
            data_y = pickle.load(input)

        data_x = np.array(data_x)
        data_x = np.nan_to_num(data_x)
        data_x = self.normalize_dataset(data_x)
        data_y = np.asarray(data_y)

        total_no_clients+=1

        counts= len(data_x)//total_no_clients
        start = client_index * counts
        end = ( client_index + 1 ) * counts

        x_train, x_test, y_train, y_test = train_test_split(data_x[start:end], data_y[start:end], test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test


# print("shape xdata:", np.shape(data_x) )
# print("shape y data:", np.shape(data_y) )
#
#
#

