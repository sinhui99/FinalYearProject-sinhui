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

def transform_to_rnn_shape(each_video_frame, video_rnn, X, y, option, n):
    each_video_frame = [int(i) for i in each_video_frame]
    
    X_rnn = []
    y_rnn = []
    frame_count = 0
    j = 0
    for i in range(video_rnn.shape[0]):
        rnn_end_frame = each_video_frame[i]
        loop_count = n[i].astype(np.int8)
        print("video ", i , " last frame of video ", rnn_end_frame)
        for k in range(20, rnn_end_frame):
            if y[k + frame_count] == 1:
                if(option == "train"):
                    for m in range(loop_count):
                        original = X[frame_count + j:frame_count + k, :]
                        noise = np.random.normal(0, .0001, original.shape)
                        new =  np.float32(original + noise)
                        X_rnn.append(new)
                        y_rnn.append(y[k + frame_count])
                X_rnn.append(X[frame_count + j:frame_count + k, :])
                y_rnn.append(y[k + frame_count])
                j+=1
            else:
                X_rnn.append(X[frame_count + j:frame_count + k, :])
                y_rnn.append(y[k + frame_count])
                j+=1
        frame_count += each_video_frame[i]
        print("total frames processed: ", frame_count)
        j=0
    
    X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)
    
    return X_rnn, y_rnn
	
video_rnn = pickle.load(open("pickle/video_rnn.pickle", "rb"))
n = pickle.load(open("pickle/n.pickle", "rb"))
each_video_frame = pickle.load(open("pickle/each_video_frame.pickle", "rb"))
X = pickle.load(open("pickle/X.pickle", "rb"))
y = pickle.load(open("pickle/y.pickle", "rb"))

X_train, y_train = transform_to_rnn_shape(each_video_frame, video_rnn, X, y, "train", n)

print(X_train.shape)
print(y_train.shape)

global X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, test_size=0.2, stratify=y_train)

print(X_train.shape)
print(X_test.shape)

with open('pickle/X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)
    
with open('pickle/X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f)
    
with open('pickle/y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)
    
with open('pickle/y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)