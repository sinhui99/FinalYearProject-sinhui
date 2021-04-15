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
		
def img_to_array(path, df_csv):
    with tf.device('/device:GPU:0'):
        if tf.test.gpu_device_name():
            print("Using GPU")
        base_model = tf.keras.applications.ResNet50(weights='imagenet', pooling='avg', include_top = False) 
        for layer in base_model.layers:
            layer.trainable = False

        list_image = []
    
        # for loop to read and store frames
        for i in tqdm(range(df_csv.shape[0])):
            # loading the image and keeping the target size as (224,224,3)
            img = image.load_img(path + df_csv['image'][i], target_size=(224,224,3))
            # converting it to array
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = base_model.predict(x)
            features = features.squeeze()

            # appending the image to the train_image list
            list_image.append(features)
          
        X = np.array(list_image)
        y = df_csv['class']
    
    return X, y
	
train = pd.read_csv('train_new.csv')
X, y= img_to_array('data/', train)

with open('pickle/X.pickle', 'wb') as f:
    pickle.dump(X, f)
    
with open('pickle/y.pickle', 'wb') as f:
    pickle.dump(y, f)