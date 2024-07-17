# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# One command line argument: name of folder where want output files to go

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
# Create my model
model = tf.keras.models.Sequential()
# Implicitely defines input layer with first hidden layer
model.add(tf.keras.layers.Dense(32, activation='softplus', input_shape=(INPUT_SIZE,)))
# Hidden layers: [32,64,128,64,32]
# Normalize batch: important for larger networks
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))
# Output layer
model.add(tf.keras.layers.Dense(OUTPUT_SIZE , activation='sigmoid')) 
# Prints out layer type, output shape, parameters, connections
model.summary()

# 3) Training
# Adam optimizer adapts learning rates for you, so no need to define a scheduler
model.compile(optimizer = 'adam', loss='mean_squared_error')
# Train the model using training data
# Capture the loss and val_loss statistics with history variable
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test,y_test))

# 4) Evaluation
# Visualize data with tensorBoard
score = model.evaluate(X_test,y_test,verbose=0)
print(f'Test loss: {score}')
model.save(sys.argv[1] + '/model1.h5')
# Create plot of error
plt.figure(1)
plt.plot(history.history['loss'], color = 'green', label = 'Training Loss')
plt.plot(history.history['val_loss'], color = 'pink', label = 'Testing Loss')
plt.title('Training versus Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(sys.argv[1] + '/error1.png')
plt.close()
# Create plot of network outputs versus actual for validation data
# TO DO: denormalize my outputs
yPredictionList = model.predict(X_test)
plt.figure(2)
plt.scatter(y_test, yPredictionList, color='blue')
plt.title('Simulated versus Interpolated Values')
plt.xlabel('Simulated')
plt.ylabel('Interpolated')
print('hello')
print(sys.argv[1] + 'simActual1.png')
plt.savefig(sys.argv[1] + '/simActual1.png')