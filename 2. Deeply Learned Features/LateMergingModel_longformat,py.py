# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:58:06 2021

@author: zy
"""

import os
import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
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
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Concatenate
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications import resnet, inception_resnet_v2, inception_v3, densenet, vgg16, vgg19
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


 


'''
X_train = np.load("train_longformat.npy")
X_test = np.load("test_longformat.npy")
y_train = np.load("y_train_long.npy")
y_test = np.load("y_test_long.npy")

y_test_short = np.load("y_test.npy")


X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.20, 
                                                      random_state=42, shuffle=True, stratify=y_train)


X_train2[:,:,:,[0]].shape

'''

X_train = np.load("train_longformat.npy")
X_test = np.load("test_longformat.npy")
y_train = np.load("y_train_long.npy")
y_test = np.load("test_longformat.npy")

y_test_short = np.load("y_test.npy")


X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.20, 
                                                      random_state=42, shuffle=True, stratify=y_train)


X_train2[:,:,:,[0]].shape


#Data Augmentation Set Up 
train_gen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='constant')


def three_inp_gen(X1, X2, X3, Y):
    genX1 = train_gen.flow(X1,Y, seed=7)
    genX2 = train_gen.flow(X2, seed=7)
    genX3 = train_gen.flow(X3, seed=7)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i, X3i], X1i[1]
            
three_inp= three_inp_gen(X_train2[:,:,:,[0]], 
                X_train2[:,:,:,[1]], 
                X_train2[:,:,:,[2]],
                y_train2)            


def metricas(y_pred, y_test_tr):
    cm = confusion_matrix(y_test_tr, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
    plt.tight_layout() 

    class_list = ['POS', 'NEG']

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks+0.5, class_list, rotation=45)
    plt.yticks(tick_marks+0.5, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test_tr,y_pred,target_names=class_list))      


#Optimizers
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=.0001)   


#DLR = 1

inp_shape= (512, 512,1)

#Firs branch

input1 = Input(inp_shape)
conv1 = layers.Conv2D(32, kernel_size=4, activation='relu')(input1)
conv2 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv1)
conv3 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv2)
pool = layers.MaxPooling2D()(conv3)  #Pooling

conv4 = layers.Conv2D(16, kernel_size=4, activation='relu')(pool)
conv5 = layers.Conv2D(16, kernel_size=4, activation='relu')(conv3)
pool2 = layers.MaxPooling2D()(conv5)  #Pooling

conv6 = layers.Conv2D(8, kernel_size=4, activation='relu')(pool2)
pool3 = layers.MaxPooling2D()(conv6)  #Pooling

# conv7 = layers.Conv2D(4, kernel_size=4, activation='relu')(pool3)
flat1 = layers.Flatten()(pool3)

#Second Branch
input2 = Input(inp_shape)
conv1_2 = layers.Conv2D(32, kernel_size=4, activation='relu')(input2)
conv2_2 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv1_2)
conv3_2 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv2_2)
pool_2 = layers.MaxPooling2D()(conv3_2)  #Pooling

conv4_2 = layers.Conv2D(16, kernel_size=4, activation='relu')(pool_2)
conv5_2 = layers.Conv2D(16, kernel_size=4, activation='relu')(conv3_2)
pool2_2 = layers.MaxPooling2D()(conv5_2)  #Pooling

conv6_2 = layers.Conv2D(8, kernel_size=4, activation='relu')(pool2_2)
pool3_2 = layers.MaxPooling2D()(conv6_2)  #Pooling

# conv7_2 = layers.Conv2D(4, kernel_size=4, activation='relu')(pool3_2)
flat2= layers.Flatten()(pool3_2)


#Third Branch
input3 = Input(inp_shape)
conv1_3 = layers.Conv2D(32, kernel_size=4, activation='relu')(input3)
conv2_3 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv1_3)
conv3_3 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv2_3)
pool_3 = layers.MaxPooling2D()(conv3_3)  #Pooling

conv4_3 = layers.Conv2D(16, kernel_size=4, activation='relu')(pool_3)
conv5_3 = layers.Conv2D(16, kernel_size=4, activation='relu')(conv3_3)
pool2_3 = layers.MaxPooling2D()(conv5_3)  #Pooling

conv6_3 = layers.Conv2D(8, kernel_size=4, activation='relu')(pool2_3)
pool3_3 = layers.MaxPooling2D()(conv6_3)  #Pooling

# conv7_3 = layers.Conv2D(4, kernel_size=4, activation='relu')(pool3_3)
flat3 = layers.Flatten()(pool3_3)

# merge feature extractors
merge = Concatenate()([flat1, flat2, flat3])

# interpretation layer
hidden1 = Dense(512, activation='relu')(merge)

# prediction output
hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(2, activation='softmax')(hidden2)

model = Model(inputs=[input1,input2, input3], outputs=output)


model.summary()

plot_model(model)
#plot_model(model,to_file="modelLM.png")   

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

NAME = 'Late Merging_LONG'

filepathdest_incep = NAME+".hdf5"

callback_setting = [ModelCheckpoint(filepath=filepathdest_incep, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)]

log_dir =  NAME  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False)

history = model.fit(three_inp, 
                    steps_per_epoch=len(X_train)/20,                
                    epochs=100,
                    verbose=1,
                    callbacks=[callback_setting, tensorboard_callback],
                    validation_data=([X_val2[:,:,:,[0]],X_val2[:,:,:,[1]],X_val2[:,:,:,[2]]],y_val2)
                   )
model.evaluate([X_test[:,:,:,[0]],
                X_test[:,:,:,[1]],
                X_test[:,:,:,[2]]],
                y_test)

Y_pred = model.predict([X_test[:,:,:,[0]],
                           X_test[:,:,:,[1]],
                           X_test[:,:,:,[2]]])

y_pred= np.argmax(Y_pred, axis =1)


y_test_tr = np.argmax(y_test, axis=1)


metricas(y_pred, y_test)
