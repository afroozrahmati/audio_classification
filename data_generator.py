from preprocessing import *
import glob
import numpy as np
import pickle
import random


preprocess = preprocessing()

pathnormal= './data/physionet/normal/'
pathabnormal = './data/physionet/abnormal/'

x_data_normal = []
y_data_normal =[]
x_data_abnormal = []
y_data_abnormal =[]

files = glob.glob(pathnormal + '/*.wav')
for file in files:
    x_data_normal.append(preprocess.extract_features(file))
    y_data_normal.append([1,0])
    print(file)

files = glob.glob(pathabnormal + '/*.wav')
for file in files:
    x_data_abnormal.append(preprocess.extract_features(file))
    y_data_abnormal.append([0,1])
    print(file)


print("shape x normal data:", np.shape(x_data_normal) )
print("shape y normal data:", np.shape(y_data_normal) )
print("shape x abnormal data:", np.shape(x_data_abnormal)  )
print("shape y abnormal data:", np.shape(y_data_abnormal)  )


data_x = np.concatenate((x_data_normal,x_data_abnormal))
data_y = np.concatenate((y_data_normal,y_data_abnormal))

# Shuffle two lists with same order
# Using zip() + * operator + shuffle()
temp = list(zip(data_x, data_y))
random.shuffle(temp)
data_x, data_y = zip(*temp)


print("shape xdata:", np.shape(data_x) )
print("shape y data:", np.shape(data_y) )


with open('physionet_MFCC80t_X.pkl', 'wb') as o:
    pickle.dump(data_x, o, pickle.HIGHEST_PROTOCOL)

with open('physionet_MFCC80t_Y.pkl', 'wb') as o:
    pickle.dump(data_y, o, pickle.HIGHEST_PROTOCOL)
