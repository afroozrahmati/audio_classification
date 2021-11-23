import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

sr = 22050  # Sampling rate
duration = 5
samples = sr * duration


# 865
def extract_features(audio_path):
    #     y, sr = librosa.load(audio_path, duration=3)
    y, sr = librosa.load(audio_path, duration=10)
    #     y = librosa.util.normalize(y)

    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)
    if len(y) > samples:  # long enough
        y = y[0:0 + samples]
    else:  # pad blank
        padding = samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, samples - len(y) - offset), 'constant')

    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048,
                                       hop_length=865,
                                       n_mels=128)
    mfccs = np.transpose(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=80))

    #     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs


pathnormal= './data/physionet/normal/'
pathabnormal = './data/physionet/abnormal/'

x_data_normal = []
y_data_normal =[]
x_data_abnormal = []
y_data_abnormal =[]

files = glob.glob(pathnormal + '/*.wav')
for file in files:
    x_data_normal.append(extract_features(file))
    y_data_normal.append([1,0])
    print(file)

files = glob.glob(pathabnormal + '/*.wav')
for file in files:
    x_data_abnormal.append(extract_features(file))
    y_data_abnormal.append([0,1])
    print(file)


print("shape x normal data:", np.shape(x_data_normal) )
print("shape y normal data:", np.shape(y_data_normal) )
print("shape x abnormal data:", np.shape(x_data_abnormal)  )
print("shape y abnormal data:", np.shape(y_data_abnormal)  )


data_x = np.concatenate((x_data_normal,x_data_abnormal))
data_y = np.concatenate((y_data_normal,y_data_abnormal))

print("shape xdata:", np.shape(data_x) )
print("shape y data:", np.shape(data_y) )

import pickle

with open('physionet_MFCC40t_X.pkl', 'wb') as o:
    pickle.dump(data_x, o, pickle.HIGHEST_PROTOCOL)

with open('physionet_MFCC40t_Y.pkl', 'wb') as o:
    pickle.dump(data_y, o, pickle.HIGHEST_PROTOCOL)
