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
		
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model
saved_model = load_model('weight_1607.hdf5')

# saved_model = tf.keras.models.load_model("model_1607")

X_testing = pickle.load(open("pickle/X_testing.pickle", "rb"))
y_testing = pickle.load(open("pickle/y_testing.pickle", "rb"))

predict = (saved_model.predict(X_testing) > 0.5).astype("int32")

# predict probabilities for test set
probability = saved_model.predict(X_testing, verbose=0)

#1607 h5py
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_testing, predict)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_testing, predict)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_testing, predict)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_testing, predict)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(y_testing, predict)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_testing, probability)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_testing, predict)
print(matrix)