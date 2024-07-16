# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 1) Preprocessing
# a) read and normalize data
# Read data columns not simVal or sitename
probCols, disCols = ['LBProb','RBProb','RTProb','LTProb'], ['d1','d2','d3','d4']
# On Frontera: /scratch1/10000/ameliakratzer14/data1c
df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/input.csv')
# Take log then normalize probability
# Access probs by doing df[proCols]
scaler = MinMaxScaler()
df[probCols] = np.log10(df[probCols])
df[probCols] = scaler.fit_transform(df[probCols])
# Normalize distance without log
df[disCols] = scaler.fit_transform(df[disCols])
# b) split data into training and testing
# X = independent variable (inputs), y = dependent variable (value to predict)
X = df.drop(columns=['simVal','interpSiteName'])
y = df['simVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Xtrain has 80 rows (since 20 rows = Xtest)and Y_train has 80 labels corresponding to 80 x samples

# 2) Network topology
# Batch size = number of samples fed to neural network at once before weights updated
BATCH_SIZE = 32
# Epochs = number of complete passes through training set - 1 epoch = about 3 batch sizes
EPOCHS = 50
INPUT_SIZE = 8
OUTPUT_SIZE = 1
# Optimize the learning rate by creating scheduler
# TO DO: play around with this to optimize my results
def scheduler(epoch):
    if epoch < 10:
        print("lr= 0.01" )
        return 0.01
    elif epoch < 40:
        print("lr= 0.001" )
        return 0.001
    elif epoch < 120:
        print("lr= 0.0001" )
        return 0.0001
    else:
        print("lr= 0.00001" )
        return 0.00001
schedule = keras.callbacks.LearningRateScheduler(scheduler)
# Create my model
model = tf.keras.models.Sequential()


# 3) Training

# 4) Evaluation