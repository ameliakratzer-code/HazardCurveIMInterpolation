# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Two command line arguments: name of folder, name of files 

# 1) Preprocessing
# a) read and normalize data
probCols = ['LBProb','RBProb','RTProb','LTProb', 'simVal']
# On Frontera: /scratch1/10000/ameliakratzer14/data1c
df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/inputa.csv')
# Take log of probabilities
df[probCols] = np.log10(df[probCols])
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()
# b) split data into training and testing
# X = independent variable (inputs), y = dependent variable (value to predict)
X = df.drop(columns=['simVal','interpSiteName'])
y = df['simVal']
X_trainU, X_testU, y_trainU, y_testU = train_test_split(X, y, test_size=0.2, random_state=42)
# Transform the data
X_train = Xscaler.fit_transform(X_trainU)
X_test = Xscaler.transform(X_testU)
y_train = Yscaler.fit_transform(y_trainU.values.reshape(-1,1)).ravel()
y_test = Yscaler.transform(y_testU.values.reshape(-1,1)).ravel()

# 2) Network topology
# Batch size = number of samples fed to neural network at once before weights updated
BATCH_SIZE = 16
# Epochs = number of complete passes through training set - 1 epoch = about 3 batch sizes
EPOCHS = 35
INPUT_SIZE = 8
OUTPUT_SIZE = 1
# Create my model
model = tf.keras.models.Sequential()
# Implicitely defines input layer with first hidden layer
model.add(tf.keras.layers.Dense(32, activation='softplus', input_shape=(INPUT_SIZE,), kernel_regularizer=tf.keras.regularizers.l2(0.0035)))
# Hidden layers: [32,64,128,64,32]
model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0035)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))
# Changed from 128 to 32 for three layers
model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.0035)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))
# Output layer
model.add(tf.keras.layers.Dense(OUTPUT_SIZE , activation='sigmoid')) 
# Prints out layer type, output shape, parameters, connections
model.summary()

# 3) Training
# Adam optimizer adapts learning rates for you, so no need to define a scheduler
optimize = tf.keras.optimizers.Adam(learning_rate=0.0016)
# 'adam'
model.compile(optimizer = optimize, loss='mean_squared_error')
# Train the model using training data
# Capture the loss and val_loss statistics with history variable
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test,y_test))

# 4) Evaluation
# Visualize data with tensorBoard
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
plt.savefig(sys.argv[1] + f'/simActual{sys.argv[2]}.png')