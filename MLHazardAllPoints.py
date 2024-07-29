import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/dataML.csv')
# Take log of probabilities
disCols = ['d1', 'd2', 'd3', 'd4']
dfRemaining = df.drop(columns=disCols+['interpsiteName'])
# Including 0 values in model for now
# Want to avoid issues with log10(0) since prob is 0 for some x values
dfRemaining = np.log10(dfRemaining + 1e-8)
dfCombined = pd.concat([dfRemaining, df[disCols], df['interpsiteName']], axis = 1)
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()

# b) split data into training and testing
# X is all probs and distances that do not start with sim
X = dfCombined.loc[:, ~dfCombined.columns.str.startswith('sim') & ~dfCombined.columns.str.startswith('interp')]
# y is all probs that start with sim
y = dfCombined.loc[:, dfCombined.columns.str.startswith('sim') | dfCombined.columns.str.startswith('interp')]
X_trainU, X_testU, y_trainU, y_testU = train_test_split(X, y, test_size=0.2, random_state=42)
testSites = y_testU['interpsiteName'].tolist()
y_trainU = y_trainU.drop(columns=['interpsiteName'])
y_testU = y_testU.drop(columns=['interpsiteName'])
# Transform the data
X_train = Xscaler.fit_transform(X_trainU)
X_test = Xscaler.transform(X_testU)
y_train = Yscaler.fit_transform(y_trainU.values.reshape(-1,51))
y_test = Yscaler.transform(y_testU.values.reshape(-1,51))

# Define the model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(32, activation='softplus', input_shape=(208,)))

model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(51 , activation='sigmoid')) 

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test,y_test))

# Evaluate the model
score = model.evaluate(X_test,y_test,verbose=0)
print(f'Test loss: {score}')
# Create plot of error
plt.figure(1)
plt.plot(history.history['loss'], color = 'green', label = 'Training Loss')
plt.plot(history.history['val_loss'], color = 'pink', label = 'Testing Loss')
plt.title('Training versus Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(sys.argv[1] + f'/error{sys.argv[2]}.png')
plt.close()
yPredictionListNorm = model.predict(X_test)
yPredictionListLog = Yscaler.inverse_transform(yPredictionListNorm.reshape(-1,51)).ravel()
yPredictionList = np.power(10, yPredictionListLog) - 1e-8
ySimListLog = Yscaler.inverse_transform(y_test.reshape(-1,51)).ravel()
ySimList = np.power(10, ySimListLog) - 1e-8
# Plot actual hazard curve versus simulated hazard curve
xValsList = [
    1.00E-04, 1.30E-04, 1.60E-04, 2.00E-04, 2.50E-04, 3.20E-04, 4.00E-04, 5.00E-04, 
    6.30E-04, 7.90E-04, 0.001, 0.00126, 0.00158, 0.002, 0.00251, 0.00316, 0.00398, 
    0.00501, 0.00631, 0.00794, 0.01, 0.01259, 0.01585, 0.01995, 0.02512, 0.03162, 
    0.03981, 0.05012, 0.0631, 0.07943, 0.1, 0.12589, 0.15849, 0.19953, 0.25119, 
    0.31623, 0.39811, 0.50119, 0.63096, 0.79433, 1, 1.25893, 1.58489, 1.99526, 
    2.51189, 3.16228, 3.98107, 5.01187, 6.30957, 7.94328, 10
]
# Hazard curve plot
print(testSites)
plt.figure(2)
plt.xscale('linear')
plt.xlim(0, 2)
plt.ylim(1e-6,1)
plt.yscale('log')
plt.xlabel('Accel (cm/s\u00B2)')
plt.ylabel('Prob (1/yr)')
plt.title('Simulated versus interpolated hazard curve')
plt.grid(axis = 'y')
plt.plot(xValsList, ySimList[:51], color='green', linewidth = 2, label = "Simulated", marker='^')
plt.plot(xValsList, yPredictionList[:51], color='pink', linewidth = 2, label = 'Interpolated', marker='^')
plt.legend()
plt.savefig(sys.argv[1] + f'/simActual{sys.argv[2]}.png')

