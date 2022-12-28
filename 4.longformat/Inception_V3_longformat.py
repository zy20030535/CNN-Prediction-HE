# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:38:32 2021

@author: zy
"""

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

X_train = np.load("train_longformat.npy")
X_test = np.load("test_longformat.npy")
y_train = np.load("y_train_long.npy")
y_test = np.load("y_test_long.npy")

y_test_short = np.load("y_test.npy")

y_test_ = [np.argmax(y_test_short[x,:]) for x in range(31)]   #

X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=True, stratify=y_train )

inp_shape = X_train2.shape[1:]


#Data Augmentation Set Up 
train_gen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='constant')

#Optimzers
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)


model_ince = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(512,512,3), pooling='max', classes=2)   ##geng gai classs


#model_incep_imnet = inception_v3.InceptionV3(include_top=False, weights='imagenet',
#                                     input_tensor=None, input_shape=(512, 512, 3), pooling='max', classes=2)  # geng gai classs
model_incep = inception_v3.InceptionV3(include_top=True, weights='imagenet')

model_incep.summary()

model_incep_imnet = inception_v3.InceptionV3(include_top=True, weights=None, input_shape=inp_shape)

'''
for new_layer, layer in zip(model_incep_imnet.layers[1:], model_incep.layers[1:]):
    new_layer.set_weights(layer.get_weights())
'''
# make a reference to VGG's input layer
inp = model_incep_imnet.input

# make a new softmax layer with num_classes neurons
new_classification_layer = layers.Dense(2, activation='softmax')    ###fei lei qi

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(model_incep_imnet.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)





model_new.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

NAME = 'IncepV3-longformat'

filepathdest_incep = NAME+".hdf5"

callback_setting = [ModelCheckpoint(filepath=filepathdest_incep, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)]

log_dir =  NAME +' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False)

history = model_new.fit(train_gen.flow(X_train2, y_train2),
                    batch_size=50,                 
                    epochs=100,
                    verbose=1,
                    callbacks=[callback_setting, tensorboard_callback],
                    validation_data=(X_val2,y_val2)
                   )

