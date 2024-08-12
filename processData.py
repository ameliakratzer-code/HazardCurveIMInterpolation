import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def makeScatterplot(ySimList, yPredictionList):
    plt.scatter(ySimList, yPredictionList, color='blue')
    plt.title('Simulated versus Interpolated Values')
    plt.xlabel('Simulated')
    plt.ylabel('Interpolated')

    model = LinearRegression()
    model.fit(ySimList.reshape(-1,1), yPredictionList)
    y_fit = model.predict(ySimList.reshape(-1,1))
    plt.plot(ySimList, y_fit, color='green', linestyle='-', label='Line of Best Fit')

    x_limits = plt.gca().get_xlim()
    y_limits = plt.gca().get_ylim()
    min_val = min(x_limits[0], y_limits[0])
    max_val = max(x_limits[1], y_limits[1])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
    plt.legend()
    plt.xlim(x_limits)
    plt.ylim(y_limits)

# Four command line arguments: input file name, name of folder, name of files, 1 if want to save

# On Frontera: /scratch1/10000/ameliakratzer14/IMMLInputs/combined_file.csv, home: /Users/ameliakratzer/Desktop/LinInterpolation/ML/IMs/allSitesIM.csv
df = pd.read_csv(sys.argv[1], low_memory = False)
# Need to drop header rows that are in middle of CSV that occured when CAT the files together
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
# Starting with 3 mil samples
df = df.iloc[:3000000]
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
print('Data cleaned')
if True:
    # Transforming inference data too
    inference_df = pd.read_csv('/scratch1/10000/ameliakratzer14/IMMLInputs/inferenceSites/USC.csv')
    X_inferenceU = inference_df.drop(columns=['IMInterp'])
    X_inference = Xscaler.transform(X_inferenceU)
    simVals = inference_df['IMInterp']

# preprocessed_data_and_scalers.pkl for first 3 mil lines (5 sites) of giant file
joblib.dump({
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'Xscaler': Xscaler,
    'Yscaler': Yscaler,
    'X_inference': X_inference,
    'simVals': simVals
}, 'preprocessed_data_and_scalers.pkl')