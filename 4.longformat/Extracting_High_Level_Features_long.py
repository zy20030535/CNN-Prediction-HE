# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 22:07:58 2021

@author: zy
"""

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
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import plot_model
import seaborn as sns

'''
incep_model = load_model("E:\ich\FinalMasterProject-AHHM-master\IncepV3-longformat.hdf5")
'''
### AlexNet
vgg19_model = load_model("E:\ich\FinalMasterProject-AHHM-master\\vgg19-Long_sgdlongformat.hdf5")


#LM_model = load_model("Late Merging=short.hdf5")


X_train = np.load("train_longformat.npy")
X_test = np.load("test_longformat.npy")
y_train_au = np.load("y_train_long.npy")
y_test_au = np.load("y_test_long.npy")


y_train =  np.load("y_train.npy")
y_test = np.load("y_test.npy")

X_train.shape

### Testing Models

def metricas(y_pred, y_test_tr):
    cm = confusion_matrix(y_test_tr, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)

    class_list = ['POS', 'NEG']

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks+0.5, class_list, rotation=45)
    plt.yticks(tick_marks+0.5, class_list, rotation=45)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test_tr,y_pred,target_names=class_list))
    
def model_converter(model):
    model_FE = Model(inputs=model.inputs, outputs= model.layers[-2].output)
    return model_FE   


def feature_extractor(X, model):
    extract_feats = list()
    for patient in range(X.shape[0]):
        img1 = X[patient]
        img_ = img1.reshape((1,512,512,3))   #(1,150,150,3)
        FE = model.predict(img_)
    
        extract_feats.append(FE[0,:])
    
        output_vector = np.asanyarray(extract_feats)
    
    return output_vector
'''
def forceshape(x):
    
    
    for i in range(X.shape[0]):
        if (i%3)==0:
            

def divide_LM_train():
    ED_list = list()
    ES_list = list()
    
    for i in np.arange(1,600,3):
        if (i % 2) == 0:
            ES_list.append(np_LM_train[i,:])
        else:
            ED_list.append(np_LM_train[i,:])
         
    ES = np.asanyarray(ES_list)
    ED = np.asanyarray(ED_list)

    return ES, ED
'''

#Incept_FE = model_converter(incep_model)
vgg19_model = model_converter(vgg19_model)
#LM_FE = model_converter(LM_model)
'''
DL_features_train = feature_extractor(X_train, Incept_FE)
#DL_features_train_ED = feature_extractor(X_ed_train, Incept_FE)

DL_features_test = feature_extractor(X_test, Incept_FE)
#DL_features_test_ED = feature_extractor(X_ed_test, Incept_FE)

DL_features_test.shape

np.savetxt('DLR_IncepModel_train.csv',
           DL_features_train, delimiter=',')
np.savetxt('DLR_IncepModel_test.csv', 
           DL_features_test, delimiter=',')
'''           
# =============================================================================
# np.savetxt('/content/gdrive/My Drive/Colab Notebooks/TFM/Extracted DLR/DLR_IncepModel_train_ED_2.csv', 
#            DL_features_train_ED, delimiter=',')
# np.savetxt('/content/gdrive/My Drive/Colab Notebooks/TFM/Extracted DLR/DLR_IncepModel_test_ED_2.csv', 
#            DL_features_test_ED, delimiter=',')
# =============================================================================
##AlexNet Normal
vgg19_DL_features_train = feature_extractor(X_train, vgg19_model)

vgg19_DL_features_test = feature_extractor(X_test, vgg19_model)


np.savetxt('DLR_vgg19_model_train.csv', vgg19_DL_features_train, delimiter=',')
np.savetxt('DLR_vgg19_model_test.csv', vgg19_DL_features_test, delimiter=',')

'''
##LM Normal
def feature_extractor_LM(X, model):
    extract_feats = list()
    for patient in range(X.shape[0]):
        
        img1 = X[patient]
        img_ = img1.reshape((1,512,512,3))    ##矩阵大小
        FE = model.predict([img_[:,:,:,[0]],
                          img_[:,:,:,[1]],
                          img_[:,:,:,[2]]])
        
        extract_feats.append(FE[0,:])
    
    output_vector = np.asanyarray(extract_feats)
    
    return output_vector


LM_features_train= feature_extractor_LM(X_train, LM_FE)

LM_features_test= feature_extractor_LM(X_test, LM_FE)

np.savetxt('DLR_LM_features_train.csv', LM_features_train, delimiter=',')
np.savetxt('DLR_LM_features_test.csv', LM_features_test, delimiter=',')

'''
