import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import csv
import joblib

# Four command line arguments: input file name, name of folder, name of files
# preprocessed_data_and_scalers.pkl on Frontera
data = joblib.load('/Users/ameliakratzer/Desktop/LinInterpolation/preprocessed_data_and_scalers.pkl')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
Xscaler = data['Xscaler']
Yscaler = data['Yscaler']
X_inference = data['X_inference']
simVals = data['simVals']
BATCH_SIZE = 800
EPOCHS = 20
INPUT_SIZE = 23
OUTPUT_SIZE = 1
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='softplus', input_shape=(INPUT_SIZE,), kernel_regularizer=tf.keras.regularizers.l2(0.005)))

model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('softplus'))

model.add(tf.keras.layers.Dense(OUTPUT_SIZE , activation='sigmoid')) 

optimize = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer = optimize, loss='mean_squared_error')
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test,y_test))

score = model.evaluate(X_test,y_test,verbose=0)
print(f'Test loss: {score}')
# Error plot
plt.figure(1)
plt.plot(history.history['loss'], color = 'green', label = 'Training Loss')
plt.plot(history.history['val_loss'], color = 'pink', label = 'Testing Loss')
plt.title('Training versus Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(sys.argv[2] + f'/error{sys.argv[3]}.png')
plt.close()
# Inference for USC only
#change to True when want it to run
if True:
    i = 0
    # Scatterplot
    yInferenceNorm = model.predict(X_inference)
    yInference = Yscaler.inverse_transform(yInferenceNorm.reshape(-1,1)).ravel()
    # Data to use for hazard curve calc
    fileName = f'USCIM.csv'
    filePath = sys.argv[2] + '/USCinference1.csv'
    with open(filePath, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(['Event', 'IMVal'])
        for IMVal in yInference:
            # Event number does not really matter
            write.writerow([f"(132, 39, {i})", IMVal])
            i += 1
plt.figure(2)
plt.scatter(simVals[:1000], yInference[:1000])
plt.savefig(sys.argv[2] + f'/simVActual{sys.argv[3]}.png')

