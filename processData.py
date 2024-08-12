import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# One command line argument = name of input file

# On Frontera: /scratch1/10000/ameliakratzer14/IMMLInputs/combined_file.csv, home: /Users/ameliakratzer/Desktop/LinInterpolation/ML/IMs/allSitesIM.csv
df = pd.read_csv(sys.argv[1], low_memory = False)
# Need to drop header rows that are in middle of CSV that occured when CAT the files together
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
# X now includes the velocity metrics
X = df.drop(columns=['IMInterp'])
y = df['IMInterp']
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()
X_trainU, X_testU, y_trainU, y_testU = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = Xscaler.fit_transform(X_trainU)
X_test = Xscaler.transform(X_testU)
y_train = Yscaler.fit_transform(y_trainU.values.reshape(-1,1)).ravel()
y_test = Yscaler.transform(y_testU.values.reshape(-1,1)).ravel()
# Transforming inference data too for USC and s505
inference_df = pd.read_csv('/scratch1/10000/ameliakratzer14/IMMLInputs/inferenceSites/USC.csv')
X_inferenceU = inference_df.drop(columns=['IMInterp'])
X_inference = Xscaler.transform(X_inferenceU)
simVals = inference_df['IMInterp']

s505_df = pd.read_csv('/scratch1/10000/ameliakratzer14/IMMLInputs/inferenceSites/s505.csv')
s505X_inferenceU = s505_df.drop(columns=['IMInterp'])
s505X_inference = Xscaler.transform(s505X_inferenceU)
s505simVals = s505_df['IMInterp']

# preprocessed_data_and_scalers.pkl for first 3 mil lines (5 sites) of giant file
# all_data_processed.pkl
joblib.dump({
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'Xscaler': Xscaler,
    'Yscaler': Yscaler,
    'X_inference': X_inference,
    'simVals': simVals,
    's505X_inference': s505X_inference,
    's505simVals': s505simVals
}, 'all_data_processed.pkl')