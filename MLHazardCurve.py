# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1) Preprocessing
# a) read and normalize data
# Read data columns not simVal or sitename
probCols, disCols = ['LBProb','RBProb','RTProb','LTProb'], ['d1','d2','d3','d4']
# On Frontera: /scratch1/10000/ameliakratzer14/data1b
df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/inputs.csv')
print(df.columns)
# Take log then normalize probability
# Access probs by doing df[proCols]
scaler = MinMaxScaler()
df[probCols] = np.log(df[probCols])
df[probCols] = scaler.fit_transform(df[probCols])
# Normalize distance without log
df[disCols] = scaler.fit_transform(df[disCols])
# b) split data into training and testing
# X = independent variable (inputs), y = dependent variable (value to predict)
X = df.drop(columns=['simVal','interpSiteName'])
y = df['simVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

# 2) Network topology

# 3) Training

# 4) Evaluation