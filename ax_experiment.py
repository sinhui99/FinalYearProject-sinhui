import cv2     # for capturing videos
import os 
import shutil
import math   # for mathematical operations
import pandas as pd
import pickle
import numpy as np   
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from glob import glob
from tqdm import tqdm
import tensorflow as tf 
from tensorflow.keras.preprocessing import image   # for preprocessing the images
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render, init_notebook_plotting
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/device:GPU:0'):
    if tf.test.gpu_device_name():
        print("GPU")
    else:
        print("no GPU")

global X_train, y_train, X_test, y_test
X_train = pickle.load(open("pickle/X_train.pickle", "rb"))
y_train = pickle.load(open("pickle/y_train.pickle", "rb"))
X_test = pickle.load(open("pickle/X_test.pickle", "rb"))
y_test = pickle.load(open("pickle/y_test.pickle", "rb"))

sizeTrain = int(X_train.shape[0]/2)
sizeTest = int(X_test.shape[0]/2)
X_train_partial = X_train[0:sizeTrain, :, :]
y_train_partial = y_train[0:sizeTrain]
print(X_train_partial.shape, " ", y_train_partial.shape)

with open('pickle/X_train_partial.pickle', 'wb') as f:
    pickle.dump(X_train_partial, f)
    
with open('pickle/y_train_partial.pickle', 'wb') as f:
    pickle.dump(y_train_partial, f)
	
X_train_partial = pickle.load(open("pickle/X_train_partial.pickle", "rb"))
y_train_partial = pickle.load(open("pickle/y_train_partial.pickle", "rb"))
print(X_train_partial.shape, " ", y_train_partial.shape)

parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.001, 0.2],
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.1, 0.5],
    },
    {
        "name": "alpha",
        "type": "range",
        "bounds": [2, 10],
        "value_type": "int"
    },
    {
        "name": "num_hidden_layers",
        "type": "range",
        "bounds": [2, 3],
        "value_type": "int"
    },
    {
        "name": "num_epochs",
        "type": "range",
        "bounds": [10, 25],
        "value_type": "int"
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [64, 128, 256],
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ['adam', 'rms', 'sgd'],
    },
]

def build_model(num_hidden_layers, alpha, dropout_rate, input_samples):
    with tf.device('/device:GPU:0'):
        if tf.test.gpu_device_name():
            units = int(input_samples / (alpha * 21))
            model = tf.keras.Sequential()
            for i in range(num_hidden_layers):
                if(i == 0):
                    model.add(LSTM(units = units, return_sequences=True, input_shape=(20, 2048)))
                    model.add(Dropout(dropout_rate))
                elif(i>0 and i<num_hidden_layers-1):
                    model.add(LSTM(units = int(2/3 * units), return_sequences=True))
                    model.add(Dropout(dropout_rate))
                else:
                    model.add(LSTM(units = int(2/3 * 2/3 * units)))
                    model.add(Dropout(dropout_rate))
            model.add(Dense(units=1, activation='sigmoid'))
            return model
        else:
            print("GPU is not available")
			
			
def training(parameterization, option, X_train, y_train, X_test, y_test):
    with tf.device('/device:GPU:0'):
        if tf.test.gpu_device_name():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            
            if (option=="tuning"):
                model = build_model(parameterization.get('num_hidden_layers'),
                                    parameterization.get('alpha'),
                                    parameterization.get('dropout_rate'),
                                   X_train.shape[0])
            else:
                model = build_model(parameterization.get('num_hidden_layers'),
                                    parameterization.get('alpha'),
                                    parameterization.get('dropout_rate'),
                                   X_train.shape[0])
            
            opt = parameterization.get('optimizer')
            opt = opt.lower()
            
            NUM_EPOCHS = parameterization.get('num_epochs')
            
            learning_rate = parameterization.get('learning_rate')
                    
            if opt == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif opt == 'rms':
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=10000,
                    decay_rate=0.96,
                    staircase=True)
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
                    
            model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[tf.keras.metrics.binary_accuracy])

            if (option == "tuning"):
                res = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'), validation_data=(X_test, y_test))
            else:
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
                mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
                res = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'), validation_data=(X_test, y_test), callbacks=[es, mcp_save])
                
            valloss = np.array(res.history['val_loss'][:])
            vallossmean = valloss.mean()
            sem = valloss.std()
            
            if np.isnan(vallossmean):
                return 9999.0, 0.0
            
            if (option == "tuning"):
                return vallossmean, sem
            else:
                return res, mcp_save, model
        else:
            print("GPU is not available")
			
init_notebook_plotting()

ax_client = AxClient()

# create the experiment.
ax_client.create_experiment(
    name="keras_experiment",
    parameters=parameters,
    objective_name='keras_cv',
    minimize=True)

def evaluate(parameters):
    return {"keras_cv": training(parameters, "tuning", X_train_partial[train_index], 
                                 y_train_partial[train_index], X_train_partial[val_index], y_train_partial[val_index])}

for i in range(25):
    skf=StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(X_train_partial.shape[0]), y_train_partial):
        print("Train index: ", train_index, " shape: ", train_index.shape, "Validation index: ", val_index, "shape: ", val_index.shape)
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
		
ax_client.save_to_json_file()

ax_client.get_trials_data_frame().sort_values('keras_cv')

best_parameters, values = ax_client.get_best_parameters()

# the best set of parameters.
for k in best_parameters.items():
    print(k)

print()

# the best score achieved.
means, covariances = values
print(means)

with open('pickle/best_parameters.pickle', 'wb') as f:
    pickle.dump(best_parameters, f)