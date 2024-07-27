import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

# Example setup
# X_train should be of shape (num_samples, num_features)
# y_train should be of shape (num_samples, 51)
df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/bigHazardModel.csv')
# Take log of probabilities
disCols = ['d1', 'd2', 'd3', 'd4']
dfRemaining = df.drop(columns=disCols)
# Including 0 values in model for now
# Want to avoid issues with log10(0) since prob is 0 for some x values
dfRemaining = np.log(dfRemaining + 1e-8)
dfCombined = pd.concat([dfRemaining, df[disCols]], axis = 1)
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()

# b) split data into training and testing
# X is all probs and distances that do not start with sim
X = df.loc[:, ~df.columns.str.startswith('sim')]
# y is all probs that start with sim
y = df.loc[:, df.columns.str.startswith('sim')]
X_trainU, X_testU, y_trainU, y_testU = train_test_split(X, y, test_size=0.2, random_state=42)
# Transform the data
X_train = Xscaler.fit_transform(X_trainU)
X_test = Xscaler.transform(X_testU)
y_train = Yscaler.fit_transform(y_trainU.values.reshape(-1,51))
y_test = Yscaler.transform(y_testU.values.reshape(-1,51))

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(208,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(51)  # Output layer for 51 probability values
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

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
yPredictionList = np.power(10, yPredictionListLog)
ySimListLog = Yscaler.inverse_transform(y_test.reshape(-1,51)).ravel()
ySimList = np.power(10, ySimListLog)
plt.figure(2)
plt.scatter(ySimList, yPredictionList, color='blue')
plt.title('Simulated versus Interpolated Values')
plt.xlabel('Simulated')
plt.ylabel('Interpolated')
plt.xscale('log')
plt.yscale('log')
plt.savefig(sys.argv[1] + f'/simActual{sys.argv[2]}.png')
plt.close()
