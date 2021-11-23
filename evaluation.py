import pickle
from preprocessing import *
import pandas as pd

# Load the Model back from file
model_Filename = "11_22_2021_16_53_02_Gamma(1)-Optim(Adam)_200.pkl"
with open(model_Filename, 'rb') as file:
    model = pickle.load(file)

validation_path='./data/physionet/validation/'
preprocess = preprocessing()
val_x = []

df = pd.read_csv(validation_path+'REFERENCE.csv',header=None)

files = glob.glob(validation_path + '/*.wav')
for file in files:
    val_x.append(preprocess.extract_features(file))
    df





