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
from sklearn.model_selection import StratifiedKFold

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/device:GPU:0'):
    if tf.test.gpu_device_name():
        print("GPU")
    else:
        print("no GPU")
		
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
                if(learning_rate >= 0.1):
                    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=learning_rate,
                        decay_steps=10000,
                        decay_rate=0.96,
                        staircase=True)
                    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
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
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
                mcp_save = ModelCheckpoint('weight_1607.hdf5', save_best_only=True, monitor='val_loss', mode='min')
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
			
global X_train, y_train, X_test, y_test
X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))
best_parameters = pickle.load(open("best_parameters.pickle", "rb"))

#1607
best_parameters['learning_rate'] = best_parameters['learning_rate']/10000
best_parameters['num_epochs'] = 500
best_parameters

#1607
history, mcp_save, best_model = training(best_parameters, "best", X_train, y_train, X_test, y_test)

best_model.summary()

#1607
import matplotlib.pyplot as plt

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, 501)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#1607
acc_train = history.history['binary_accuracy']
acc_val = history.history['val_binary_accuracy']
epochs = range(1, 501)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

best_model.save("model_1607")