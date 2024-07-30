import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Create model is called 51 times total, and it will interpolate the value at Xn for the 20 test sites
# 1st run = have the interpolated points for x0 for the 20 test sites
def createModel(xtr, xte, ytr, yte):
    X_train = Xscaler.fit_transform(xtr)
    X_test = Xscaler.transform(xte)
    # Still reshape to 1 since only 1 output
    y_train = Yscaler.fit_transform(ytr.values.reshape(-1,1)).ravel()
    y_test = Yscaler.transform(yte.values.reshape(-1,1)).ravel()
    BATCH_SIZE = 16
    EPOCHS = 35
    # Input size varies depending if is edge point - 13 compared to 18
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = 1
    # Create my model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='softplus', input_shape=(INPUT_SIZE,), kernel_regularizer=tf.keras.regularizers.l2(0.0035)))
    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0035)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('softplus'))
    model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.0035)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('softplus'))
    model.add(tf.keras.layers.Dense(OUTPUT_SIZE , activation='sigmoid')) 
    optimize = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = optimize, loss='mean_squared_error')
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test,y_test))
    # Evaluation
    score = model.evaluate(X_test,y_test,verbose=0)
    print(f'Test loss: {score}')
    plt.figure()
    plt.plot(history.history['loss'], color = 'green', label = 'Training Loss')
    plt.plot(history.history['val_loss'], color = 'pink', label = 'Testing Loss')
    plt.title('Training versus Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Only arg is name of folder on Desktop
    os.makedirs(path + f'/{sys.argv[1]}')
    plt.savefig(path + f'/{sys.argv[1]}/error.png')
    plt.close()





    # Create plot of network outputs versus actual for validation data
    yPredictionListNorm = model.predict(X_test)
    yPredictionListLog = Yscaler.inverse_transform(yPredictionListNorm.reshape(-1,1)).ravel()
    yPredictionList = np.power(10, yPredictionListLog)
    ySimListLog = Yscaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    ySimList = np.power(10, ySimListLog)
    plt.figure(2)
    plt.scatter(ySimList, yPredictionList, color='blue')
    plt.title('Simulated versus Interpolated Values')
    plt.xlabel('Simulated')
    plt.ylabel('Interpolated')
    plt.xscale('log')
    plt.yscale('log')
    X = np.log10(ySimList).reshape(-1, 1)
    y = np.log10(yPredictionList)
    model = LinearRegression()
    model.fit(X, y)
    y_fit = model.predict(X)
    # Plot line of best fit
    plt.plot(ySimList, np.power(10, y_fit), color='green', linestyle='-', label='Line of Best Fit')
    plt.savefig(path + f'/{sys.argv[1]}/simActual.png')

# Read in all the data for the 51 networks
df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/dataML.csv')
# Normalize all the data before splitting it up
disCols = ['d1', 'd2', 'd3', 'd4']
dfRemaining = df.drop(columns=disCols+['interpsiteName'])
dfRemaining = np.log10(dfRemaining + 1e-8)
dfCombined = pd.concat([dfRemaining, df[disCols], df['interpsiteName']], axis = 1)
# Want interpolated data for all x values for same test sites so train test split first
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()
path = '/Users/ameliakratzer/Desktop'
X = dfCombined.loc[:, ~dfCombined.columns.str.startswith('sim') & ~dfCombined.columns.str.startswith('interp')]
y = dfCombined.loc[:, dfCombined.columns.str.startswith('sim') | dfCombined.columns.str.startswith('interp')]
X_trainU, X_testU, y_trainU, y_testU = train_test_split(X, y, test_size=0.2, random_state=42)
testSites = y_testU['interpsiteName'].tolist()
y_trainU = y_trainU.drop(columns=['interpsiteName'])
y_testU = y_testU.drop(columns=['interpsiteName'])
# Split up train and test arrays by x value
for x in range(51):
    # Only have one neighbor
    if x == 0:
        X_trainUS = X_trainU[[f'LB{x+1}', f'LB{x}', 'd1', f'RB{x+1}', f'RB{x}', 'd2', f'RT{x+1}', f'RT{x}', 'd3', f'LT{x+1}', f'LT{x}', 'd4', f'simVal{x+1}']]
        X_testUS = X_testU[[f'LB{x+1}', f'LB{x}', 'd1', f'RB{x+1}', f'RB{x}', 'd2', f'RT{x+1}', f'RT{x}', 'd3', f'LT{x+1}', f'LT{x}', 'd4', f'simVal{x+1}']]
        y_trainUS = y_trainU[f'simVal{x}']
        y_testUS = y_testU[f'simVal{x}']
    elif x == 50:
        X_trainUS = X_trainU[[f'LB{x-1}', f'LB{x}', 'd1', f'RB{x-1}', f'RB{x}', 'd2', f'RT{x-1}', f'RT{x}', 'd3', f'LT{x-1}', f'LT{x}', 'd4', f'simVal{x-1}']]
        X_testUS = X_testU[[f'LB{x-1}', f'LB{x}', 'd1', f'RB{x-1}', f'RB{x}', 'd2', f'RT{x-1}', f'RT{x}', 'd3', f'LT{x-1}', f'LT{x}', 'd4', f'simVal{x-1}']]
        y_trainUS = y_trainU[f'simVal{x}']
        y_testUS = y_testU[f'simVal{x}']
    else:
        X_trainUS = X_trainU[[f'LB{x-1}', f'LB{x+1}', f'LB{x}', 'd1', f'RB{x-1}', f'RB{x+1}', f'RB{x}', 'd2', f'RT{x-1}', f'RT{x+1}', f'RT{x}', 'd3', f'LT{x-1}', f'LT{x+1}', f'LT{x}', 'd4', f'simVal{x-1}', f'simVal{x+1}']]
        X_testUS = X_testU[[f'LB{x-1}', f'LB{x+1}', f'LB{x}', 'd1', f'RB{x-1}', f'RB{x+1}', f'RB{x}', 'd2', f'RT{x-1}', f'RT{x+1}', f'RT{x}', 'd3', f'LT{x-1}', f'LT{x+1}', f'LT{x}', 'd4', f'simVal{x-1}', f'simVal{x+1}']]
        y_trainUS = y_trainU[f'simVal{x}']
        y_testUS = y_testU[f'simVal{x}']
    createModel(X_trainUS, X_testUS, y_trainUS, y_testUS)