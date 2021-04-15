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

def count_video_frame_num(video_name):  
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length
    
def exclude_suspense_frame(length,j,k,sus_frame):
    if k == len(sus_frame[j]) - 1:
        non_sus_start_frame = sus_frame[j][k][1] + 1
        non_sus_end_frame = length[j]
    else:
        non_sus_start_frame = sus_frame[j][k][1] + 1
        non_sus_end_frame = sus_frame[j][k+1][0] - 1
    return non_sus_start_frame, non_sus_end_frame
    
def read_txt_file(filename):
    f = open(filename, "r")
    temp = f.read()
    video_path = temp.split('\n')
    
    pd_sus = pd.DataFrame()
    pd_sus['video_path'] = video_path
    pd_sus = pd_sus[:]
    
    video_name = []
    s_label = []
    ns_label = []
    for i in range(pd_sus.shape[0]):
        video_name.append(pd_sus['video_path'][i].split('/')[1])
        s_label.append("suspense")
        ns_label.append("non-suspense")
        
    pd_sus['video_name'] = video_name
    pd_non_sus = pd_sus.copy()
    
    pd_sus['label'] = s_label
    pd_non_sus['label'] = ns_label
    
    video_rnn = pd.DataFrame(pd_sus.sort_values(by = "video_name"))
    video_rnn = video_rnn.reset_index(drop = True)
    
    return pd_sus, pd_non_sus, video_rnn
    
def cal_start_end_frame(txtfilefolder, videofolder, pd_sus, pd_non_sus):
    # open the .txt file which have suspense label
    sus_frame = []
    video_text_file = []
    total_frame_per_video = []
    for i in range(pd_sus.video_name.shape[0]):
        if("mkv" in pd_sus.video_name[i]):
            video_text_file.append(pd_sus.video_name[i].replace("mkv", "txt"))
        elif ("mp4" in pd_sus.video_name[i]):
            video_text_file.append(pd_sus.video_name[i].replace("mp4", "txt"))
        else:
            video_text_file.append(pd_sus.video_name[i].replace("webm", "txt"))
        f = open(txtfilefolder + video_text_file[i], "rt")
        sus_frame.append([[int(token) for token in line.split()] for line in f.readlines()[::]])
        total_frame_per_video.append(count_video_frame_num(videofolder + pd_sus.video_name[i]))
        
    sus_start_frame = []
    sus_end_frame = []
    non_sus_start_frame = []
    non_sus_end_frame = []
    non_sus_start_frame_each = []
    non_sus_end_frame_each = []
    sus_scene_per_video = []
    non_sus_scene_per_video = []
    
    for j in range(len(sus_frame)):
        non_sus_scene_count = 0
        sus_scene_per_video.append(len(sus_frame[j]))
        for k in range(len(sus_frame[j])):
            if k == 0 and sus_frame[j][k][0] != 0:
                non_sus_start_frame.append(0)
                non_sus_end_frame.append(sus_frame[j][k][0] - 1)
                non_sus_scene_count += 1
            non_sus_start_frame_each, non_sus_end_frame_each = exclude_suspense_frame(total_frame_per_video,j,k,sus_frame)
            sus_start_frame.append(sus_frame[j][k][0])
            sus_end_frame.append(sus_frame[j][k][1])
            non_sus_start_frame.append(non_sus_start_frame_each)
            non_sus_end_frame.append(non_sus_end_frame_each)
            non_sus_scene_count += 1
        if non_sus_scene_count == 0:
            non_sus_start_frame.append(0)
            non_sus_end_frame.append(count_video_frame_num(videofolder + pd_sus.video_name[j]))
            non_sus_scene_per_video.append(1)
        else:
            non_sus_scene_per_video.append(non_sus_scene_count)
    
    pd_sus['scene_per_video'] = sus_scene_per_video
    pd_non_sus['scene_per_video'] = non_sus_scene_per_video
    pd_sus = pd_sus.loc[pd_sus.index.repeat(pd_sus.scene_per_video)].reset_index(drop=True)
    pd_non_sus = pd_non_sus.loc[pd_non_sus.index.repeat(pd_non_sus.scene_per_video)].reset_index(drop=True)
    pd_sus['start_frame'] = sus_start_frame
    pd_sus['end_frame'] = sus_end_frame
    pd_non_sus['start_frame'] = non_sus_start_frame
    pd_non_sus['end_frame'] = non_sus_end_frame
    data = pd.concat([pd_sus, pd_non_sus], ignore_index=True)
    
    return data
    
def extract_frames(path, data):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # storing the frames from training videos
    for i in tqdm(range(data.shape[0])):
        count = 0
        currentframe = 0
        # Read the video from specified path 
        cam = cv2.VideoCapture(data.video_path[i]) 
        frameRate = cam.get(5) #frame rate
        
        try: 
            # creating a folder named data 
            if not os.path.exists(path): 
                os.makedirs(path) 
        
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 
            
        # frame 
        currentframe = data.start_frame[i]
        cam.set(1, currentframe)
        while(currentframe <= data.end_frame[i]): 
            
            # reading from frame
            ret,frame = cam.read()
            
            if (ret != True):
                break
                
            if math.floor(currentframe) % math.floor(frameRate) == 0:
                if("mkv" in data.video_name[i]):
                    name = path + '/' + data.label[i] + '_' + data.video_name[i].replace(".mkv", "_") + str(currentframe) + '.jpg'
                elif("mp4" in data.video_name[i]):
                    name = path + '/' + data.label[i] + '_' + data.video_name[i].replace(".mp4", "_") + str(currentframe) + '.jpg'
                else:
                    name = path + '/' + data.label[i] + '_' + data.video_name[i].replace(".webm", "_") + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame)
                
            currentframe += 1
        
        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows() 
 
def calculate_index(df_data):
    index = []
    for i in range(len(df_data)):
        if "_" in df_data.iloc[i,2]:
            index.append(df_data.iloc[i, 0].split('_')[3].split('.')[0])
        else:
            index.append(df_data.iloc[i, 0].split('_')[2].split('.')[0])
    index = [int(i) for i in index]
    return index
    
def cal_each_video_frame(df_data):
    video = df_data.iloc[0,2]
    each_video_frame = []
    each_video_extra_frame = []
    ct = 0
    suspense = 0
    non_suspense = 0
    for i in range(df_data.shape[0]):
        ct += 1
        if df_data.iloc[i, 1] == 1:
            suspense += 1
        else:
            non_suspense += 1
        if video != df_data.iloc[i, 2]:
            each_video_frame.append(ct - 1)
            if df_data.iloc[i, 1] == 1:
                each_video_extra_frame.append(non_suspense - (suspense - 1))
                suspense = 1
                non_suspense = 0
            else:
                each_video_extra_frame.append((non_suspense - 1) - suspense)
                suspense = 0
                non_suspense = 1
            
            video = df_data.iloc[i, 2]    
            ct = 1
        if i == (df_data.shape[0] -1):
            each_video_frame.append(ct)
            each_video_extra_frame.append(non_suspense -  suspense)
    return each_video_frame, each_video_extra_frame
    
def save_frame_to_csv(path, csv_name):
    # getting the names of all the images
    images = glob(path + "/*.jpg")
    list_image = []
    list_class = []
    list_video_name = []
    for i in tqdm(range(len(images))):
        # creating the image name
        list_image.append(images[i].split('/')[1])
        # creating the class of image
        if (images[i].split('/')[1].split('_')[0] == 'non-suspense'):
            list_class.append(0)
        else:
            list_class.append(1)
        if "XLWx0_I1qLQ" in images[i].split('/')[1] or "_y3rFsvz8qQ" in images[i].split('/')[1]:
            temp = "_".join(images[i].split('/')[1].split('_')[1:3])
            list_video_name.append(temp)
        else:
            list_video_name.append(images[i].split('/')[1].split('_')[1])
        
    # storing the images and their class in a dataframe
    df_data = pd.DataFrame()
    df_data['image'] = list_image
    df_data['class'] = list_class
    df_data['video'] = list_video_name
    df_data['index'] = calculate_index(df_data)
    
    df_data = df_data.sort_values(by = ['video', 'index'], ascending = True)
    
    each_video_frame, each_video_extra_frame = cal_each_video_frame(df_data)
    
    df_data = df_data.reset_index(drop = True)
    df_data = df_data.drop(['index'], axis = 1)
    df_data = df_data.drop(['video'], axis = 1)
    df_data.to_csv(csv_name,header=True, index=False)
    
    return each_video_frame, each_video_extra_frame
    
def calculate_for_loop_no(each_video_frame, each_video_extra_frame):
    each_video_frame = [int(i) for i in each_video_frame]
    each_video_extra_frame = [int(i) for i in each_video_extra_frame]
    each_video_frame = np.array(each_video_frame, dtype=float)
    each_video_extra_frame = np.array(each_video_extra_frame)
    sus_count = (each_video_frame - each_video_extra_frame) / 2
    for_loop_num = np.divide(each_video_extra_frame, sus_count, out=np.zeros_like(each_video_frame), where=sus_count!=0)
    return (np.floor(for_loop_num))
    
pd_sus, pd_non_sus, video_rnn = read_txt_file("trainlist01.txt")
train = cal_start_end_frame("textfiles/", "videos/", pd_sus, pd_non_sus)

extract_frames('data', train)

each_video_frame, each_video_extra_frame = save_frame_to_csv('data', 'train_new.csv')
n = calculate_for_loop_no(each_video_frame, each_video_extra_frame)

if os.path.exists('pickle'):
        shutil.rmtree('pickle')
try: 
    # creating a folder named data 
    if not os.path.exists('pickle'): 
        os.makedirs('pickle') 
        
    # if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
    
with open('pickle/video_rnn.pickle', 'wb') as f:
    pickle.dump(video_rnn, f)
    
with open('pickle/n.pickle', 'wb') as f:
    pickle.dump(n, f)
    
with open('pickle/each_video_frame.pickle', 'wb') as f:
    pickle.dump(each_video_frame, f)
    
with open('pickle/each_video_extra_frame.pickle', 'wb') as f:
    pickle.dump(each_video_extra_frame, f)