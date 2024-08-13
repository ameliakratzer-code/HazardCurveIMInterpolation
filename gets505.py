import numpy as np
import tensorflow as tf
import joblib
import csv

model = tf.keras.models.load_model('/scratch1/10000/ameliakratzer14/ML2Outputs/modelgpuallb.h5')
data = joblib.load('/scratch1/10000/ameliakratzer14/all_data_processed.pkl')
Yscaler = data['Yscaler']
s505X_inference = data['s505X_inference']
s505simVals = data['s505simVals']
i = 0
yInferenceNorm = model.predict(s505X_inference)
yInference = Yscaler.inverse_transform(yInferenceNorm.reshape(-1,1)).ravel()
# Data to use for hazard curve calc
filePath = '/scratch1/10000/ameliakratzer14/ML2Outputs/s505inference.csv'
with open(filePath, 'w', newline='') as file:
    write = csv.writer(file)
    write.writerow(['Event', 'IMVal'])
    for IMVal in yInference:
        # Event number does matter for hazard curve calc code so need to use correctEvent.py to get
        write.writerow([f"(132, 39, {i})", IMVal])
        i += 1