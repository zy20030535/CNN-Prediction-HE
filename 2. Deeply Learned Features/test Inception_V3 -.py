# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:33:36 2021

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



#train_dir = r'D:\github resource\ESCC_ML-master\example\images'
#test_dir = r'C:\Users\zy\Desktop\testing'

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


def seeimage(sel_image):
    print('start   seeimage')
    data_1 =sel_image
    fig = plt.figure(figsize=(15,6))
    fig.add_subplot(131)
    plt.imshow(data_1)
    plt.axis('off')
    plt.title('Cine-MRI')
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
    plt.show()

def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    segPath = [os.path.join(path,i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg
# read regions of interest (ROI) in Nifti format 
def loadImgArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    imgPath = [os.path.join(path,i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)    
    return img

# Feature Extraction
#Cropping box
def maskcroppingbox(images_array, use2D=False):
    print('start   maskcroppingbox')
    images_array_2 = np.argwhere(images_array)
    print(images_array_2,images_array_2.shape)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    print((zstart, ystart, xstart, zstop, ystop, xstop))
    return (zstart, ystart, xstart), (zstop, ystop, xstop)
        
def ROI_extraction(image_array,mask_array,fileway):
    print('start  ROI_extraction')
    #image_array = sitk.GetArrayFromImage(imageFilepath) 
    #mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    
    #print(zstart, ystart, xstart,zstop, ystop, xstop)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    
    roi_images1 = zoom(roi_images, zoom=[224/roi_images.shape[0], 224/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=np.float)    
    seeimage(roi_images1)
    out = sitk.GetImageFromArray(roi_images2)
    path = fileway +'path_saved.nii'
    sitk.WriteImage(out, path)

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

            self.dataset['img'].append(os.path.join(file[0],selname(file,'im')))
            #self.dataset['msk_es'].append(os.path.join(file[0], selname(file,'mask')))
            self.dataset['mask'].append(os.path.join(file[0], selname(file,'seg1')))
            #self.dataset['msk_es'].append(os.path.join(file[0], selname(file,'mask')))

            patient_id = os.path.basename(file[0])
            search_string = os.path.join(path, selname(file,'im') )  #what is this for really?
            #print('search_string',search_string)
            
            '''
            self.search_string_fr_ed.append(os.path.join(path,  patient_id,  file[2][-4]))
            self.search_string_msk_ed.append(os.path.join(path,  patient_id , file[2][-3]))
            self.search_string_fr_es.append(os.path.join(path,  patient_id ,  file[2][-2]))
            self.search_string_msk_es.append(os.path.join(path,  patient_id ,  file[2][-1]))
            '''
            self.search_string_img.append(os.path.join(path,  patient_id,  selname(file,'im')))
            self.search_string_mask.append(os.path.join(path,  patient_id , selname(file,'seg1')))
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
                '''
                cont += 1
                if cont==3:
                    classs = line.lstrip("Group: ")
                    classs = classs.rstrip("\n")
                    class_list.append(classs)
                '''    
                classs = myline.lstrip("Group: ")
                classs = classs.rstrip("\n")
                class_list.append(classs)     
                print(classs)
    
    y_array1 = LabelEncoder().fit_transform(class_list)
    y = to_categorical(y_array1, num_classes=2)    #you ji ge fenlei  num_classes=?
    
    return y


def show_img(data):
    print('start  show_img')
    for i in range(data.shape[0]):
        io.imshow(data[i,:,:], cmap = 'gray')
        #print(i)
        io.show()
        
        
def choose_channal(array,channel):
    print('start choose_channal')
    x=channel
    #print('images_array_2',images_array_2,images_array_2.shape)
    for y in range(array.shape[0]):
        if y!=x:
            for a in range(array.shape[1]):
                for b in range(array.shape[2]):
                    array[y,a,b]=0
    print('choose_channal',array.shape)
    #show_img(array[x:,:,])
    return array
        

        
def get_dataset_short_maxchannal(search_dir):
    ''' Returns each structure in the the center frame as a channel
     X_ed = (150,150, 3)
     X_es = (150,150,3)
     X_tot = (150, 150, 3) '''
   # shape = (512, 512)

    dataset = Dataset(search_dir)
    print("dataset",dataset)
    final_array = []
    #for x in range(len(dataset.patient_ids)):
    for x in range(len(dataset.patient_ids)):
        print('patient_ids',dataset.patient_ids[x],dataset.search_string_mask[x])
        mask_file = sitk.ReadImage(dataset.search_string_mask[x])
        #print('mask_file',mask_file)
        
        mask = sitk.GetArrayFromImage(mask_file)
        print('mask_array.shape',mask.shape)
        #show_img(mask)
        #array2 = np.array(mask.dataobj)
        #print('mask_array.shape',array2.shape)
        mask_nb = nib.load(dataset.search_string_mask[x])
         
        array_nb = np.array(mask_nb.dataobj)
        #print('mask_array.shape',array2.shape)
        #array_nb = np.array(array_nb.dataobj)
        channels = max_channal(array_nb)
        
        mask=choose_channal(mask,channels)
        #show_img(mask)
        #sel_channels = round((low_channels+up_channels)/ 2)
        print('sel_channels',channels)
        #print('mask',mask)
        #array_2 = mask[channels-1:channels+1:, :, ]
        #array_3 = mask[0:, :, ]
        #print('array_3',array_3)
        
        
        
        #         array_2 = MinMaxScaled(array_2)
        #array_22 = cv2.resize(array_2, shape, interpolation=cv2.INTER_CUBIC)
        #array_22 = array_2 # Frame ES
        #print('mask_array.shape',array_22.shape)
        #show_img(array_22)
        img_file = sitk.ReadImage(dataset.search_string_img[x])
        
        img = sitk.GetArrayFromImage(img_file)
        print('img_array.shape',img.shape)
        #show_img(img)
        #array = np.array(f_ed.dataobj)
        #array = np.array(img.dataobj)
        #print('img_array.shape',array.shape)    
        #array_1 = array[:, :, channels-1]
        #print('array1.shape',array_1.shape)
        #seeimage(img,array_2)
        
        
        
        #         array_1 = MinMaxScaled(array_1)
        #array_11 = cv2.resize(array_1, shape, interpolation=cv2.INTER_CUBIC)  # Frame ED
        #seeimage(array_11)
        
        ROI= ROI_extraction(img,mask,dataset.search_string_img[x])
        #seeimage
        #ROI=tran_channal(ROI)

        print('ROI.shape',ROI.shape)
        #io.imshow(ROI)
        
        #lved = MinMaxScaled(np.multiply(array_11, np.equal(array_22, 1)))
        
        #print('lved',lved.shape)
       # print("Patient: ", x, "Equal 1", "Max ", np.max(lved), "Min ", np.min(lved))

        #myoed = MinMaxScaled(np.multiply(array_11, np.equal(array_22, 2)))
        #print("Patient: ", x, "Equal 2", "Max ", np.max(myoed), "Min ", np.min(myoed))
       # print('lved',lved.shape)
        #rved = MinMaxScaled(np.multiply(array_11, np.equal(array_22, 3)))
        #print("Patient: ", x, "Equal 3", "Max ", np.max(rved), "Min ", np.min(rved))
        '''    
        ###es list
        
        
        f_es = nib.load(dataset.search_string_fr_es[x])

        array3 = np.array(f_es.dataobj)
        array_3 = array3[:, :, [sel_channels - 1]]
        #         array_3 = MinMaxScaled(array_3)
        array_33 = cv2.resize(array_3, shape, interpolation=cv2.INTER_CUBIC)

        msk_es = nib.load(dataset.search_string_msk_ed[x])

        array4 = np.array(msk_es.dataobj)
        array_4 = array4[:, :, [sel_channels - 1]]
        #         array_4 = MinMaxScaled(array_4)
        array_44 = cv2.resize(array_4, shape, interpolation=cv2.INTER_CUBIC)

        es_ = np.multiply(array_33, array_44)
        #         es_input = np.resize(es_, (256, 256,1))

        lves = MinMaxScaled(np.multiply(array_33, np.equal(array_44, 1)))
        myoes = MinMaxScaled(np.multiply(array_33, np.equal(array_44, 2)))
        rves = MinMaxScaled(np.multiply(array_33, np.equal(array_44, 3)))
        '''    
        # #         plt.imshow(lves[:,:,0])
        # #         plt.show()
        # print("Patient: ", x, "Equal 1", "Max ", np.max(lves), "Min ", np.min(lves))
        # #         plt.imshow(myoes[:,:,0])
        # #         plt.show()
        # print("Patient: ", x, "Equal 2", "Max ", np.max(myoes), "Min ", np.min(myoes))
        # #         plt.imshow(rves[:,:,0])
        # #         plt.show()
        # print("Patient: ", x, "Equal 3", "Max ", np.max(rves), "Min ", np.min(rves))

        # final = np.dstack((array_1, array_2, array_3, array_4))  #Frame and Mask without modification
        # final2 = np.dstack((array_11, array_22, array_33, array_44))   #Frame and Mask resized (not normalized)
        # final3 = np.dstack((array_11, array_22, ed_, array_33, array_44, es_))   #frame mask, multiplication
        #final4 = np.dstack((lved, myoed, rved, lves, myoes, rves))
        final4 = np.dstack(ROI)
        print('final4',final4.shape,np.min(final4), np.max(final4))
        final_array.append(final4)
        #print('final_array',final_array)

    X_set = np.asanyarray(final_array)

    X_ed = X_set
    print("X_ED array Shape ", X_ed.shape)
    print("X_ED - Min {} - Max {}".format(np.min(X_ed), np.max(X_ed)))
    '''
    X_es = X_ed
    print("X_ES array Shape ", X_es.shape)
    print("X_ES - Min {} - Max {}".format(np.min(X_es), np.max(X_ed)))
    '''    
    return X_set


train=get_dataset_short_maxchannal(train_dir)
test=get_dataset_short_maxchannal(test_dir)

#train=train.reshape(inp_shape,3)

np.save('train.npy',train)
np.save('test.npy',test)

X_train = train
#X_train_ED = test[2]
X_test = test
#X_test_ED = te




## Concat the arrays

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
y_test=get_classes(test_dir)

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

'''
model_ince = inception_v3.InceptionV3(include_top=False, weights='imagenet', 
                                      input_tensor=None, input_shape=(512,512,3), pooling='max', classes=2)   ##geng gai classs
'''
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

NAME = 'IncepV3-short-Imagenet'

filepathdest_incep = NAME+".hdf5"

callback_setting = [ModelCheckpoint(filepath=filepathdest_incep, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)]

log_dir =  NAME +' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False)

history = model_new.fit(train_gen.flow(X_train2, y_train2),
                    batch_size=50,                 
                    epochs=800,
                    verbose=1,
                    callbacks=[callback_setting, tensorboard_callback],
                    validation_data=(X_val2,y_val2)
                   )



incep_model = load_model('C:/Users/zy/Desktop/FinalMasterProject-AHHM-master/test/2. Deeply Learned Features\IncepV3-short-Imagenet.hdf5')


loss, accuracy = incep_model.evaluate(X_test,y_test)

Y_pred = incep_model.predict(X_test)
y_pred = np.argmax(Y_pred, axis =1)

y_test_tr= np.argmax(y_test, axis=1)


def metricas(y_pred, y_test_tr):
    cm = confusion_matrix(y_test_tr, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
    plt.tight_layout() 

    class_list = ['POS','NEG']

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks+0.5, class_list, rotation=45)
    plt.yticks(tick_marks+0.5, class_list, rotation=45, va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test_tr,y_pred,target_names=class_list))

metricas(y_pred, y_test_tr)



