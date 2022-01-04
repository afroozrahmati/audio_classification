from preprocessing import *
import glob
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class clients_data_generation:
    def __init__(self):
        pass

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

    def target_distribution(self,q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


    def get_clients_data(self,client_name):
        preprocess = preprocessing()
        pathnormal = './data/physionet/flower/'+client_name+'/normal/'
        pathabnormal = './data/physionet/flower/'+client_name+'/abnormal/'

        x_data_normal = []
        y_data_normal = []
        x_data_abnormal = []
        y_data_abnormal = []

        files = glob.glob(pathnormal + '/*.wav')
        for file in files:
            x_data_normal.append(preprocess.extract_features(file))
            y_data_normal.append([1, 0])
            print(file)

        files = glob.glob(pathabnormal + '/*.wav')
        for file in files:
            x_data_abnormal.append(preprocess.extract_features(file))
            y_data_abnormal.append([0, 1])
            print(file)

        print("shape x normal data:", np.shape(x_data_normal))
        print("shape y normal data:", np.shape(y_data_normal))
        print("shape x abnormal data:", np.shape(x_data_abnormal))
        print("shape y abnormal data:", np.shape(y_data_abnormal))

        data_x = np.concatenate((x_data_normal, x_data_abnormal))
        data_y = np.concatenate((y_data_normal, y_data_abnormal))

        print("shape xdata:", np.shape(data_x))
        print("shape y data:", np.shape(data_y))
        return data_x,data_y
