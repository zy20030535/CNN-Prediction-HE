# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:37:03 2021

@author: zy
"""

#!/usr/bin/env python
# coding: utf-8

# ## Deep Learning Testing

# In[5]:


import os
import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import cv2
import re
import os
import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import time
import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
from keras.preprocessing import image
import matplotlib.pyplot as plt

## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Activation
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications import resnet, inception_resnet_v2, inception_v3, densenet, vgg16, vgg19
from collections import Counter
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from numpy import moveaxis
from numpy import asarray
from PIL import Image
import skimage.io as io


train_dir = r'C:/Users/zy/Desktop/source_img'


    
    

def findsel_channal(np_mask):
    print('start    findsel_channal')
    data=np_mask
    p=data.shape[2]
    print(p)
    a=0
    b=0
    for x in range(p):
        #print(data_2[:,:x])
        if  np.sum(data[:,:,x])>0:
            a=x
            break
     
    for y in range(1,p):
        #print(data_2[:,:,p-y])
        if  np.sum(data[:,:,p-y])>0:
            b=p-y+1
            break
    
    #print(a,b)
    return a,b

def max_channal(np_mask):
    print('start    max_channal')
    data=np_mask
    p=data.shape[2]
    #print(p)
    a=[]
    b=0
    for x in range(p):
        b=np.sum(data[:,:,x])
        #print(b)
        if np.sum(data[:,:,x])==0.0:
            b=0
        #print(b)
        a.append(b)
        #print(a)
    a=np.array(a)
    a=np.argmax(a)
    return a

def selname(file,target_file_name):
    
    for a in range(len(file[2])):
        datanames = file[2][a]
        #print("datanames",datanames)
        pattern = re.compile(target_file_name)
        match = pattern.search(datanames)
        if match !=None:
            #print('return',file[2][a]) 
            return file[2][a]


def seeimage(sel_image,path):
    print('start   seeimage')
    data_1 =sel_image
    fig = plt.figure(figsize=(15,6))
    fig.add_subplot(131)
#   plt.imshow(data_1,cmap='gray')
#   plt.axis('off')
#   plt.title(path)
#   plt.show()
    '''
    fig.add_subplot(132)
    plt.imshow(data_2[:,:,0] )
    plt.axis('off')
    plt.title('Segmentation-Mask')
    fig.add_subplot(133)
    plt.imshow(np.multiply(data_1[:,:,0],data_2[:,:,0]))
    plt.axis('off')
    plt.title('MRI-Mask')
    '''  

# Feature Extraction
#Cropping box
def maskcroppingbox(images_array, use2D=False):
    print('start   maskcroppingbox')
    images_array_2 = np.argwhere(images_array)
    print(images_array_2,images_array_2.shape)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    print((zstart, ystart, xstart, zstop, ystop, xstop))
    return (zstart, ystart, xstart), (zstop, ystop, xstop)
        
def ROI_extraction(image_array,mask_array,patient_ids):
    print('start  ROI_extraction')
    #image_array = sitk.GetArrayFromImage(imageFilepath) 
    #mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    
    #print(zstart, ystart, xstart,zstop, ystop, xstop)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    
    roi_images1 = zoom(roi_images, zoom=[512/roi_images.shape[0], 512/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=np.float)    
#   seeimage(roi_images1[:,:,0],patient_ids)
#   out = sitk.GetImageFromArray(roi_images2)
    
#   NAME='C:/Users/zy/Desktop/PATH/training/'
#    path =NAME+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+patient_ids+'path_saved.nii'
#   sitk.WriteImage(out, path)

    return roi_images2
    '''
    x = image.img_to_array(roi_images2)
    print('shape', x.shape)
    
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    base_model_pool_features = model.predict(x)
    
    feature_map = base_model_pool_features[0]
    feature_map = feature_map.transpose((2,1,0))
    features = np.max(feature_map,-1)
    features = np.max(features,-1)
    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(features):
    	deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures
    '''
    

def tran_channal(img_array):
     print('start  tran_channal')
      # load the color image
     
      # convert to numpy array
     data = asarray(img_array)
      
      # change channels last to channels first format
     #data = moveaxis(data, 2, 0)
     print(data.shape)
      # change channels first to channels last format
     data = moveaxis(data, 0, 2)
     print(data.shape)
     return data

class Dataset():

    def __init__(self, path, counter_max=0, type='4D'):

        self.path = path
        self.class_folders = [folder for folder in os.listdir(self.path) if 'class' in folder]
        #self.dataset = {'img_filenames': [], 'msk_ed': [], 'msk_es': []}
        self.dataset = {'img_filenames': [],'img': [], 'mask': []}
        # self.es_frames = pd.read_excel(os.path.join(self.path,"ES_frames.xlsx"))
        
        self.patient_ids = []
        
        '''
        self.search_string_fr_ed = []
        self.search_string_msk_ed = []
        self.search_string_fr_es = []
        self.search_string_msk_es = []
        '''
        self.search_string_img=[]
        self.search_string_mask=[]
        
        # img_path = os.path.join(path,'image')
        # seg_path = os.path.join(path,'segs')

        counter = 0

        for file in os.walk(path):

            if counter_max != 0 and counter > counter_max:
                break

            if file[0] == path:
                continue

            #             file[2].sort()
            #             if ".DS_Store" in file[2]:
            #                 file[2].remove(".DS_Store")

            self.dataset['img'].append(os.path.join(file[0],selname(file,'img')))
            #self.dataset['msk_es'].append(os.path.join(file[0], selname(file,'mask')))
            self.dataset['mask'].append(os.path.join(file[0], selname(file,'mask')))
            #self.dataset['msk_es'].append(os.path.join(file[0], selname(file,'mask')))

            patient_id = os.path.basename(file[0])
            search_string = os.path.join(path, selname(file,'img') )  #what is this for really?
            #print('search_string',search_string)
            
            '''
            self.search_string_fr_ed.append(os.path.join(path,  patient_id,  file[2][-4]))
            self.search_string_msk_ed.append(os.path.join(path,  patient_id , file[2][-3]))
            self.search_string_fr_es.append(os.path.join(path,  patient_id ,  file[2][-2]))
            self.search_string_msk_es.append(os.path.join(path,  patient_id ,  file[2][-1]))
            '''
            self.search_string_img.append(os.path.join(path,  patient_id,  selname(file,'img')))
            self.search_string_mask.append(os.path.join(path,  patient_id , selname(file,'mask')))
            #self.search_string_fr_es.append(os.path.join(path,  patient_id ,  file[2][-2]))
            #self.search_string_msk_es.append(os.path.join(path,  patient_id ,  file[2][-1]))


            
            #     pdb.set_trace()
            image_location = glob.glob(search_string)

            self.patient_ids.append(patient_id)

            self.dataset['img_filenames'].append(image_location)

            #print('self.datasetimg_filenames',patient_id)

def MinMaxScaled(x):
    xnew= ((x-np.min(x))/(np.max(x)-np.min(x)))
    return xnew

def get_classes(path):
    print('start  get_classes')
    class_list = []

   # classs= []
    for file in os.walk(path):
        cont = 0
        if file[0] == path:
            continue
        with open ((os.path.join(file[0], selname(file,'Info.cfg')))) as myfile:
            for myline in myfile:
                cont += 1
                if cont == 1:
                    classs = myline.lstrip("Group: ")
                    classs = classs.rstrip("\n")
                    class_list.append(classs)
                    print(classs)
    
    y_array1 = LabelEncoder().fit_transform(class_list)
    y = to_categorical(y_array1, num_classes=2)    #you ji ge fenlei  num_classes=?
    
    return y






# ----------------------------------------------------------------------------


# Lets apply the cv2 resizing

#test=get_dataset_short(train_dir)















def y_class_long(y):
    list_y = []
    for i in range(len(y)):
        for x in range(0, 3):
            list_y.append(y[i])

    return np.asanyarray(list_y)

#print(test.shape)



import os
import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import time

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Activation
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications import resnet, inception_resnet_v2, inception_v3, densenet, vgg16, vgg19
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



#%load_ext tensorboard
'''
X_train_ES = np.load("/content/gdrive/My Drive/Colab Notebooks/TFM/Data/Short Format/X_es.npy")
X_train_ED = np.load("/content/gdrive/My Drive/Colab Notebooks/TFM/Data/Short Format/X_ed.npy")
X_test_ES = np.load("/content/gdrive/My Drive/Colab Notebooks/TFM/Data/Short Format/X_es_test.npy")
X_test_ED = np.load("/content/gdrive/My Drive/Colab Notebooks/TFM/Data/Short Format/X_ed_test.npy")
y_train = np.load("/content/gdrive/My Drive/Colab Notebooks/TFM/Data/Short Format/y_train.npy")
y_test = np.load("/content/gdrive/My Drive/Colab Notebooks/TFM/Data/Short Format/y_test.npy")
'''



train = np.load('train_longformat.npy')
#test = np.load('test_longformat.npy')
X_train = train
#X_train_ED = test[2]
#X_test = test
#X_test_ED = test[2]



  

#X_train = np.concatenate([X_train_ES, X_train_ED],axis=0)
print(X_train.shape)
#print(X_test.shape)

import pandas as pd
'''
train_df = pd.read_csv('/home/zy/code/FinalMasterProject-AHHM-master/ACDC_(Radiomics+Clinical)_Training.csv')
print(train_df.shape)
test_df = pd.read_csv('/home/zy/code/FinalMasterProject-AHHM-master/ACDC_(Radiomics+Clinical)_Training.csv')
print(test_df.shape)
y_train =  train_df['class']
y_test = test_df['class']
y_train = np.concatenate([y_train, y_train],axis=0)
print(y_train.shape)
'''


y_train=get_classes(train_dir)
#y_test=get_classes(test_dir)


#y_train = np.concatenate([train, train],axis=0)   ##
#print(y_train.shape)


np.save('y_train.npy', y_train)
#np.save('y_test.npy', y_test)

#y_train = np.concatenate([train, train],axis=0)   ##
#print(y_train.shape)
y_train_long=y_class_long(y_train)
#y_test_long=y_class_long(y_test)

np.save('y_train_long.npy',y_train_long)
#np.save('y_test_long.npy',y_test_long)


#y_test=y_train


# =============================================================================
# 
# 
# X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.20,  shuffle=True, stratify=y_train )
# 
# inp_shape = X_train2.shape[1:]
# 
# 
# #Data Augmentation Set Up 
# train_gen = ImageDataGenerator(rotation_range=40,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              zoom_range=0.2,
#                              horizontal_flip=True,
#                              fill_mode='constant')
# 
# 
# #Optimzers
# sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
# adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# 
# 
# model = Sequential()
# 
# model.add(Conv2D(96, (11, 11), strides = (4,4), padding='valid', activation='relu', input_shape = (512,512,3)))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
# 
# model.add(Conv2D(256, (5, 5), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
# 
# model.add(Conv2D(384, (3, 3), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
# model.add(Conv2D(384, (3, 3), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
# model.add(Conv2D(256, (3, 3), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
# 
# model.add(Dropout(0.20))
# 
# model.add(Flatten())
# 
# model.add(Dense(9216, activation='relu'))
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(2, activation='softmax'))    ###   Dense(2  gai cheng fen lei shuzi 
# 
# model.summary()
# 
# alex_model=model
# 
# alex_model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
# 
# NAME = 'AlexNet-Short_sgd'
# 
# #filepathdest_incep = "/content/gdrive/My Drive/Colab Notebooks/TFM/Models/"+NAME+".hdf5"
# 
# callback_setting = [ModelCheckpoint(filepath='first.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)]
# 
# log_dir = "logs" + NAME + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False)
# 
# csv_logger = CSVLogger('first.3.log')
# earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='auto')
# reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='auto')
# 
# history = alex_model.fit(train_gen.flow(X_train2, y_train2),
#                     batch_size=64,                 
#                     epochs=400,
#                     verbose=1,
#                     callbacks=[callback_setting,reduceLR, earlystopping,tensorboard_callback],
#                     #callbacks=[csv_logger, callback_setting],
#                     validation_data=(X_val2,y_val2)
#                    )
# 
# 
# alex_model = load_model('C:/Users/zy/Desktop/FinalMasterProject-AHHM-master/test/2. Deeply Learned Features/first.3.01-0.59.hdf5')
# 
# 
# #loss, accuracy = alex_model.evaluate(X_test_ES,y_test)
# 
# loss, accuracy = alex_model.evaluate(X_test,y_test)
# 
# #loss, accuracy = alex_model.evaluate(X_train,y_train)
# 
# #loss, accuracy = alex_model.evaluate(X_test_ED,train)
# 
# Y_pred = alex_model.predict(X_test)
# 
# 
# #Y_pred = alex_model.predict(X_train)
# 
# 
# y_pred= np.argmax(Y_pred, axis =1)
# '''
# Y_pred_ed = alex_model.predict(X_test_ES)
# y_pred_ed = np.argmax(Y_pred_ed, axis =1)
# 
# #y_test_tr = np.argmax(y_test, axis=1)
# '''
# y_test_tr = np.argmax(y_test, axis=1)
# 
# #y_test_tr = np.argmax(y_train, axis=1)
# 
# 
# def metricas(y_pred, y_test_tr):
#     cm = confusion_matrix(y_test_tr, y_pred)
#     
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 
#     plt.figure(figsize=(10,6))
#     plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix")
#     sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
#     plt.tight_layout() 
# 
#     class_list = ['POS','NEG']
# 
#     tick_marks = np.arange(len(class_list))
#     plt.xticks(tick_marks+0.5, class_list, rotation=45)
#     plt.yticks(tick_marks+0.5, class_list, rotation=45, va='center')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
# 
#     print(classification_report(y_test_tr,y_pred,target_names=class_list))
#     
#     
#     
# metricas(y_pred, y_test_tr)
# 
# 
# alex_model = load_model('first.3.01-0.59.hdf5')   ###hd5f file
# 
# 
# loss, accuracy = alex_model.evaluate(X_test,y_test)
# 
# Y_pred = alex_model.predict(X_test)
# 
# y_pred= np.argmax(Y_pred, axis =1)
# '''
# Y_pred_ed = alex_model.predict(X_test_ES)
# y_pred_ed = np.argmax(Y_pred_ed, axis =1)
# 
# #y_test_tr = np.argmax(y_test, axis=1)
# '''
# y_test = np.argmax(y_test, axis=1)
# 
# 
# metricas(y_pred, y_test)
# =============================================================================


