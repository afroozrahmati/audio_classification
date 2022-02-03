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
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np
import math

fs = 1000
## Order of five works well with ECG signals
cutoff_high = 0.5
cutoff_low = 2
powerline = 60
order = 5

## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a
## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a
def notch_filter(cutoff, q):
    nyq = 0.5*fs
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a

def highpass(data, fs, order=5):
    b,a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b,a,data)
    return x

def lowpass(data, fs, order =5):
    b,a = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(b,a,data)
    return y

def notch(data, powerline, q):
    b,a = notch_filter(powerline,q)
    z = lfilter(b,a,data)
    return z

def final_filter(data, fs, order=5):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order = order)
    y = lfilter(d, c, x)
    f, e = notch_filter(powerline, 30)
    z = lfilter(f, e, y)
    return z




class preprocessing:
    def __init__(self,sr = 22050,duration = 5):
          # Sampling rate
        self.duration = duration
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


    # hop_lenght =865 for duartion=5
    # hop_lenght = 520 for duration=3
    # hop_lenght = 1725 for duration =10
    def extract_features(self,audio_path,hop_lenght,features,n_mels):
        #     y, sr = librosa.load(audio_path, duration=3)
        y, sr = librosa.load(audio_path, duration = self.duration)
        #     y = librosa.util.normalize(y)
        # y= self.band_pass_filter(y)
        # self.samples = fs * self.duration

        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)
        if len(y) > self.samples:  # long enough
            y = y[0:0 + self.samples]
        else:  # pad blank
            padding = self.samples - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, self.samples - len(y) - offset), 'constant')

        S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048,
                                           hop_length=hop_lenght,
                                           n_mels=n_mels)
        mfccs = np.transpose(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=features))
        #mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
        #     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return mfccs

    def band_pass_filter(self,ecg_singal):
        filter_signal = final_filter(ecg_singal, fs, order)
        return filter_signal



    #rename files to the proper format, so we can partition it for each client
    def rename_files(self,path):
        files = glob.glob(path + '/*.wav')
        i = 0
        for file in files:
            new_name='.'+file.split('.')[1]+'-'+str(i)+'.wav'
            i+=1
            os.rename(file, new_name)

    #load related data for each client
    def load_data(self,pathnormal,pathabnormal,client_index, total_no_clients,features,timesteps):
        #duration = 5 and librosa sample rate is 22050 then the formula for hop_lenght is
        hop_lenght= math.ceil((5 * 22050) / timesteps)
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
                x_data_normal.append(self.extract_features(file,hop_lenght,features,timesteps))
                y_data_normal.append([1,0])
                print(file)


        files = glob.glob(pathabnormal + '/*.wav')
        counts= len(files)//total_no_clients
        start = client_index * counts
        end = ( client_index + 1 ) * counts
        for file in files:
            if start <= int(file.split('-')[1].split('.')[0]) < end:
                x_data_abnormal.append(self.extract_features(file,hop_lenght,features,timesteps))
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

        with open('./data/processed_files/augmented_data_40_2_X.pkl', 'wb') as o:
            pickle.dump(data_x, o, pickle.HIGHEST_PROTOCOL)

        with open('./data/processed_files/augmented_data_40_2_Y.pkl', 'wb') as o:
            pickle.dump(data_y, o, pickle.HIGHEST_PROTOCOL)


    def load_processed_data(self,file_path_x,file_path_y):
        with open(file_path_x, 'rb') as input:
            data_x = pickle.load(input)

        with open(file_path_y, 'rb') as input:
            data_y = pickle.load(input)

        data_x = np.array(data_x)
        data_x = np.nan_to_num(data_x)
        data_x = self.normalize_dataset(data_x)
        data_y = np.asarray(data_y)

        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def load_processed_train_data(self,total_no_clients):
        with open('./data/processed_files/augmented_data_40_2_X.pkl', 'rb') as input:
            data_x = pickle.load(input)

        with open('./data/processed_files/augmented_data_40_2_Y.pkl', 'rb') as input:
            data_y = pickle.load(input)

        #total_no_clients+=1
        counts= len(data_x)//total_no_clients
        start = 0
        end =  (total_no_clients -1 )  * counts

        data_x = np.array(data_x)
        data_x = np.nan_to_num(data_x)
        data_x = self.normalize_dataset(data_x)
        data_y = np.asarray(data_y)

        x_train, x_test, y_train, y_test = train_test_split(data_x[start:end], data_y[start:end], test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def load_processed_partition(self,client_index, total_no_clients,file_path_x,file_path_y):
        with open(file_path_x, 'rb') as input:
            data_x = pickle.load(input)

        with open(file_path_y, 'rb') as input:
            data_y = pickle.load(input)

        total_no_clients += 1

        counts= len(data_x)//total_no_clients
        start = client_index * counts
        end = ( client_index + 1 ) * counts

        data_x = np.array(data_x)
        data_x = np.nan_to_num(data_x)
        data_x = self.normalize_dataset(data_x)
        data_y = np.asarray(data_y)

        x_train, x_test, y_train, y_test = train_test_split(data_x[start:end], data_y[start:end], test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test



