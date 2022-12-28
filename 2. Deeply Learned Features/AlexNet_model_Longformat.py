# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:46:36 2021

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
from keras.layers import Conv3D, MaxPooling3D, Dense, BatchNormalization, Dropout, Flatten, Activation
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
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications import resnet, inception_resnet_v2, inception_v3, densenet, vgg16, vgg19
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

X_train = np.load("train_longformat.npy")
X_test = np.load("test_longformat.npy")
y_train = np.load("y_train_long.npy")
y_test = np.load("y_test_long.npy")

y_test_short = np.load("y_test.npy")

y_test_ = [np.argmax(y_test_short[x,:]) for x in range(31)]   #

X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.20,
                                                      random_state=42, shuffle=True, stratify=y_train)

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


#alex_model = load_model("AlexNet-Short_sgd.hdf5")
alex_model = Sequential()


# 1st Convolutional Layer
alex_model.add(Conv2D(filters=96, input_shape=inp_shape, kernel_size=(
    11, 11), strides=(4, 4), padding="valid", activation='relu'))
# Max Pooling
alex_model.add(MaxPooling2D(pool_size=(
    2, 2), strides=(2, 2), padding="valid"))

# 2nd Convolutional Layer
alex_model.add(Conv2D(filters=256, kernel_size=(11, 11),
                      strides=(1, 1), padding="valid", activation='relu'))
# Max Pooling
alex_model.add(MaxPooling2D(pool_size=(
    2, 2), strides=(2, 2), padding="valid"))

# 3rd Convolutional Layer
alex_model.add(Conv2D(filters=384, kernel_size=(3, 3),
                      strides=(1, 1), padding="valid", activation='relu'))

# 4th Convolutional Layer
alex_model.add(Conv2D(filters=384, kernel_size=(3, 3),
                      strides=(1, 1), padding="valid", activation='relu'))

# 5th Convolutional Layer
alex_model.add(Conv2D(filters=256, kernel_size=(3, 3),
                      strides=(1, 1), padding="valid", activation='relu'))
# Max Pooling
alex_model.add(MaxPooling2D(pool_size=(
    2, 2), strides=(2, 2), padding="valid"))

# Passing it to a Fully Connected layer
alex_model.add(Flatten())
# 1st Fully Connected Layer
alex_model.add(Dense(4096, activation='relu'))
# Add Dropout to prevent overfitting
alex_model.add(Dropout(0.5))

# 2nd Fully Connected Layer
alex_model.add(Dense(4096, activation='relu'))
alex_model.add(Activation("relu"))
# Add Dropout
alex_model.add(Dropout(0.5))

# Output Layer
alex_model.add(Dense(2, activation='softmax'))


alex_model.compile(loss='categorical_crossentropy',
                   optimizer=adam,
                   metrics=['accuracy'])
alex_model.summary()



alex_model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
alex_model.summary()
 






#alex_model = model.summary()

NAME = 'AlexNet-Long_sgd'

filepathdest_incep = NAME+"longformat"+".hdf5"

callback_setting = [ModelCheckpoint(filepath=filepathdest_incep, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)]

log_dir = "longformat"+NAME +' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False)

history = alex_model.fit(train_gen.flow(X_train2, y_train2),
                    batch_size=50,                 
                    epochs=100,
                    verbose=1,
                    callbacks=[callback_setting, tensorboard_callback],
                    validation_data=(X_val2,y_val2)
                   )

loss, accuracy = alex_model.evaluate(X_test,y_test)

Y_pred = alex_model.predict(X_test)
y_pred = np.argmax(Y_pred, axis =1)

y_test_tr = np.argmax(y_test, axis=1)


cm = confusion_matrix(y_test_tr, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()

class_list = ['POS', 'NEG']

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
plt.tight_layout()

tick_marks = np.arange(len(class_list))
plt.xticks(tick_marks, class_list, rotation=45)
plt.yticks(tick_marks, class_list)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

def most_frequent(List): 
    return max(set(List), key = List.count) 

y_au_pred = alex_model.predict(X_test)

final_result = list()
for patient in np.arange(0,300,6):
  list_1 = list()
  for slice_ in range(6):
      list_1.append(np.argmax(y_au_pred[(patient+slice_):(slice_+1+patient), :]))
      print((np.argmax(y_au_pred[(patient+slice_):(slice_+1+patient), :])))
  print('Most Frequent', most_frequent(list_1))
  print('-------------------------------------')
  print('Patient  ', patient)
  final_result.append(most_frequent(list_1))
  
def checking_up_results(model, X_test):
    y_pred = model.predict(X_test)
    final_result = list()
    for patient in np.arange(0,300,6):
      list_1 = list()
      for slice_ in range(6):
          list_1.append(np.argmax(y_pred[(patient+slice_):(slice_+1+patient), :]))
          # print((np.argmax(y_pred[(patient+slice_):(slice_+1+patient), :])))
      # print('Most Frequent', most_frequent(list_1))
      # print('-------------------------------------')
      # print('Patient  ', patient)
      final_result.append(most_frequent(list_1))

    return final_result

y_true_list = [np.argmax(y_test[x, :]) for x in range(3)]

final_result = checking_up_results(alex_model, X_test)

print(accuracy_score(y_test_, final_result))

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
    plt.yticks(tick_marks+0.5, class_list, rotation=45, va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test_tr,y_pred,target_names=class_list))
    
metricas(final_result, y_test_)    
