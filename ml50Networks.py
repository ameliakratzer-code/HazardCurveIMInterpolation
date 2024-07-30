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
def createModel(xtr, xte, ytr, yte, x):
    global plotDict
    X_train = Xscaler.fit_transform(xtr)
    X_test = Xscaler.transform(xte)
    # Still reshape to 1 since only 1 output
    y_train = Yscaler.fit_transform(ytr.values.reshape(-1,1)).ravel()
    y_test = Yscaler.transform(yte.values.reshape(-1,1)).ravel()
    BATCH_SIZE = 16
    EPOCHS = 35
    # Input size varies depending if is edge point - 13 compared to 18
    INPUT_SIZE = X_train.shape[1]
    print(f'Input size: {INPUT_SIZE}')
    OUTPUT_SIZE = 1
    # Create my model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='softplus', input_shape=(INPUT_SIZE,)))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('softplus'))
    model.add(tf.keras.layers.Dense(32))
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
    directory = path + f'/{sys.argv[1]}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path + f'/{sys.argv[1]}/error{x}.png')
    plt.close()
    # Add vals to dictionary for hazard curve plots
    yPredictionListNorm = model.predict(X_test)
    yPredictionListLog = Yscaler.inverse_transform(yPredictionListNorm.reshape(-1,1)).ravel()
    yPredictionList = np.power(10, yPredictionListLog)
    ySimListLog = Yscaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    ySimList = np.power(10, ySimListLog)
    # Key = sitename, value = list with two sublists - first sublist has sim val, second sublist
    for i, key in enumerate(plotDict):
        if i < 20:
            plotDict[key][0].append(ySimList[i])
            plotDict[key][1].append(yPredictionList[i])

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
X = dfCombined.loc[:, ~dfCombined.columns.str.startswith('interp')]
y = dfCombined.loc[:, dfCombined.columns.str.startswith('sim') | dfCombined.columns.str.startswith('interp')]
X_trainU, X_testU, y_trainU, y_testU = train_test_split(X, y, test_size=0.2, random_state=42)
testSites = y_testU['interpsiteName'].tolist()
plotDict = {key: [[], []] for key in testSites}
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
    createModel(X_trainUS, X_testUS, y_trainUS, y_testUS, x)
# Time to plot the hazard curves using my dict
xValsList = [
    1.00E-04, 1.30E-04, 1.60E-04, 2.00E-04, 2.50E-04, 3.20E-04, 4.00E-04, 5.00E-04, 
    6.30E-04, 7.90E-04, 0.001, 0.00126, 0.00158, 0.002, 0.00251, 0.00316, 0.00398, 
    0.00501, 0.00631, 0.00794, 0.01, 0.01259, 0.01585, 0.01995, 0.02512, 0.03162, 
    0.03981, 0.05012, 0.0631, 0.07943, 0.1, 0.12589, 0.15849, 0.19953, 0.25119, 
    0.31623, 0.39811, 0.50119, 0.63096, 0.79433, 1, 1.25893, 1.58489, 1.99526, 
    2.51189, 3.16228, 3.98107, 5.01187, 6.30957, 7.94328, 10
]
for key in plotDict:
    plt.figure()
    plt.xscale('linear')
    plt.xlim(0, 2)
    plt.ylim(1e-6,1)
    plt.yscale('log')
    plt.xlabel('Accel (cm/s\u00B2)')
    plt.ylabel('Prob (1/yr)')
    plt.title(f'Hazard curves {key}, 2 sec RotD50')
    plt.grid(axis = 'y')
    plt.plot(xValsList, plotDict[key][0], color='green', linewidth = 2, label = "Simulated", marker='^')
    plt.plot(xValsList, plotDict[key][1], color='pink', linewidth = 2, label = 'Interpolated', marker='^')
    plt.legend()
    plt.savefig(path + f'/{sys.argv[1]}/{key}.png')
    plt.close()
print('Plots plotted')