import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import csv
import joblib
import pandas as pd

def makeInferencePlot(x, y, name):
    plt.figure()
    plt.scatter(x, y)
    plt.title('Simulated versus Interpolated Values')
    plt.xlabel('Simulated')
    plt.ylabel('Interpolated')

    model = LinearRegression()
    model.fit(np.array(x).reshape(-1,1), y)
    y_fit = model.predict(np.array(x).reshape(-1,1))
    plt.plot(np.array(x), y_fit, color='green', linestyle='-', label='Line of Best Fit')

    x_limits = plt.gca().get_xlim()
    y_limits = plt.gca().get_ylim()
    min_val = min(x_limits[0], y_limits[0])
    max_val = max(x_limits[1], y_limits[1])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
    plt.legend()
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.savefig(sys.argv[2] + f'/{name}.png')
    plt.close()

# Three command line arguments: input file name, name of folder, name of files
# preprocessed_data_and_scalers.pkl on Frontera, USC_preprocessed_data.pkl for USC
# /scratch1/10000/ameliakratzer14/all_data_processed.pkl for all data
data = joblib.load('/scratch1/10000/ameliakratzer14/all_data_processed.pkl')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
Xscaler = data['Xscaler']
Yscaler = data['Yscaler']
X_inference = data['X_inference']
simVals = data['simVals']
s505X_inference = data['s505X_inference']
s505simVals = data['s505simVals']
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
model.save(sys.argv[2] + f'/model{sys.argv[3]}.h5')

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
    filePath = sys.argv[2] + f'/USCinference{sys.argv[3]}.csv'
    with open(filePath, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(['Event', 'IMVal'])
        for IMVal in yInference:
            # Event number does matter for hazard curve calc code so need to use correctEvent.py to get
            write.writerow([f"(132, 39, {i})", IMVal])
            i += 1

# Plot inference sights
x = simVals[:]
y = yInference[:]
s505x = s505simVals
s505y = s505X_inference
USCPlotName = 'USCInference.png'
s505PlotName = 's505Inference.png'
makeInferencePlot(x, y, USCPlotName)
makeInferencePlot(s505x, s505y, s505PlotName)


