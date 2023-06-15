# -*- coding: utf-8 -*-
"""
Created on Tue May 23 23:19:51 2023

@author: zy
"""

from skimage import exposure
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
import torch

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
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import numpy as np
import nibabel as nib
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
data = r'C:/Users/zy/Desktop/source_img'


    
    

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


def D3_to_2d(img):

    
    # 将图像从3D降到2D
    data_2d = np.squeeze(img)
    
    # 显示灰阶图像
#    plt.imshow(data_2d, cmap='gray')
#   plt.show()

    print("D3_to_2d",data_2d.shape)
    return data_2d


def D2_TO_RGB(IMG):

    
    # 假设图像为cv2.CV_64F的灰阶图像
    gray_data = IMG
    gray_data = np.float32(gray_data)
    
    # 如果图像为64位浮点数，将其转换为8位无符号整数类型
    if gray_data.dtype == np.float64:
        gray_data = (gray_data * 255).astype(np.uint8)
    else:
        # 如果图像类型为其他类型，可以根据需要进行相应的转换
        pass
    
    # 将图像从灰阶格式转换为RGB格式
    rgb_data = cv2.cvtColor(gray_data, cv2.COLOR_GRAY2RGB)
    rgb_image = rgb_data.astype(np.float32)


    rgb_image /= 255.0

# 将值裁剪到合适的范围内
    rgb_data = np.clip(rgb_image, 0, 1)


    # 显示转换后的RGB图像
#   plt.imshow(rgb_data)
#   plt.axis('off')
#   plt.show()
    return rgb_data




def seeimage(sel_image):
    print('start   seeimage')
    data_1 =sel_image
    fig = plt.figure(figsize=(512,512))
    fig.add_subplot(131)
    plt.imshow(data_1,cmap='gray')
    plt.axis('off')
    plt.show()
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
        
def ROI_extraction_3(image_array,mask_array,patient_ids):
    print('start  ROI_extraction')
    #image_array = sitk.GetArrayFromImage(imageFilepath) 
    #mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    
    #print(zstart, ystart, xstart,zstop, ystop, xstop)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    
    roi_images1 = zoom(roi_images, zoom=[512/roi_images.shape[0], 512/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=np.float)    
#   seeimage(roi_images1[:,:,0])
    gry_data=D3_to_2d(roi_images2)
    roi_rgb=D2_TO_RGB(gry_data)

    print("roi_image.shape")

    return roi_rgb
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
import matplotlib.pyplot as plt
import nibabel as nib
def seeimg3d(img):
    # 加载灰度图像
    img = img

    # 显示中间切片的所有 slice
    n = img.shape[2]
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(5*n, 5))
    for i in range(n):
        ax[i].imshow(img[:, :, i], cmap='gray', extent='auto')
        ax[i].axis('off')
    plt.show()

from skimage.filters import rank
from skimage.morphology import binary_dilation, remove_small_objects
from skimage.segmentation import clear_border
def resize_max_img(niputimg):
    # 加载灰度图像
    img = niputimg
    from skimage.measure import label
    from skimage.measure import regionprops
    selem = np.ones((3,3,3))
    img = exposure.rescale_intensity(img, out_range=(-1, 1))
    # 对全局像素值进行阈值分割
    threshold = rank.otsu(img,selem)
    binary = img > threshold
    # 计算适当的 buffer_size 值
    buffer_size = min(binary.shape) // 10  # 缩小 20 倍
    # 对二值化后的图像进行膨胀
    dilated = binary_dilation(binary)

    # 去除小噪声
    cleaned = remove_small_objects(dilated, min_size=10)

    # 去除边缘噪声
    trimmed = clear_border(cleaned, buffer_size=buffer_size)

    # 获取有意义部分的坐标位置
    indices = np.where(trimmed)

    x_min, x_max = np.min(indices[0]), np.max(indices[0])
    y_min, y_max = np.min(indices[1]), np.max(indices[1])
    z_min, z_max = np.min(indices[2]), np.max(indices[2])

    # 截取有意义部分
    cropped_img = img[x_min:x_max, y_min:y_max, z_min:z_max]

    # 截取有意义部分
    cropped_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
    print("此图像最大经线：",cropped_img.shape)
    return cropped_img

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
                if cont==1:
                    classs = myline.lstrip("Group: ")
                    classs = classs.rstrip("\n")
                    class_list.append(classs)
#                   class_list.append(classs)
                '''    
                classs = myline.lstrip("Group: ")
                classs = classs.rstrip("\n")
                class_list.append(classs)     
                print(classs)
                '''    
#   y_array1 = LabelEncoder().fit_transform(class_list)
#   y = to_categorical(y_array1, num_classes=2)    #you ji ge fenlei  num_classes=?
    
    return class_list


def show_img(data):
    print('start  show_img')
    for i in range(data.shape[0]):
        io.imshow(data[i,:,:], cmap = 'gray')
        #print(i)
        io.show()
         
        
def choose_channal(array,channel):
    print('start choose_channal')
    x=channel-1
    #print('images_array_2',images_array_2,images_array_2.shape)
    for y in range(array.shape[0]):
        if y!=x:
            for a in range(array.shape[1]):
                for b in range(array.shape[2]):
                    array[y,a,b]=0
    print('choose_channal',array.shape)
    #show_img(array[x:,:,])
    return array
def tran_RGB_XYZD(inputarray):    
    img = inputarray
    
    # 将原图像数组添加一个额外的RGB通道维度，形成一个形状为（x，y，z，3）的新数组
    img_rgb = img[:, :, :, np.newaxis]
    
    # 获取新数组的造型
    print(img_rgb.shape)  
    return  img_rgb 

def tran_RGB_XYZD_3D(inputarray):   
    # 一个向量为（x，y，z）的三维图像数组
    img = inputarray

    # 将原图像数组添加一个额外的RGB通道维度，形成一个形状为（x，y，z，3）的新数组
    img_rgb = np.repeat(img[:, :, :, np.newaxis], 3, axis=-1)

    # 获取新数组的造型
    return  img_rgb 


def tran_RGB_XY1_RGB_XY3(inputarray):
    # 一个向量为（x，y，z）的三维图像数组
    img = inputarray

    # 将原图像数组添加一个额外的RGB通道维度，形成一个形状为（x，y，z，3）的新数组
    img_rgb = np.tile(img, (1, 1, 3))

    # 获取新数组的造型
    return img_rgb

def resize_128_128_16(input_img):

    img = input_img

    h, w, d = (128, 128, 15)

    resized_img = np.zeros((h, w, d))

    for i in range(d):
        # 从原图像中提取一个二维层面
        slice_img = img[:, :, i, 0]

        # 进行比例缩放
        resized_slice_img = cv2.resize(
            slice_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 将缩放后的层面保存到目标3D数组
        resized_img[:, :, i] = resized_slice_img

# 将3D图像转换成4D图像，以便于后续处理
    resized_img = resized_img[:, :, :, np.newaxis]

# 保存成比例压缩后的图像
    return resized_img














def get_dataset_long(search_dir):
    '''Format corresponds to the end wanted
    0 = ED  . 1 = ES ,  2 = Both  '''
    shape = (512, 512)

    dataset = Dataset(search_dir)

    final_array2 = []
    #for x in range(len(dataset.patient_ids)):
    for x in range(len(dataset.patient_ids)):
        print('patient_ids',dataset.patient_ids[x],dataset.search_string_mask[x])
        mask_file = sitk.ReadImage(dataset.search_string_mask[x])
        #print('mask_file',mask_file)

        maskimg = sitk.GetArrayFromImage(mask_file)

        print('mask_array.shape', maskimg.shape)
        #show_img(mask)
        #array2 = np.array(mask.dataobj)
        #print('mask_array.shape',array2.shape)
        mask_nb = nib.load(dataset.search_string_mask[x])

        array_nb = np.array(mask_nb.dataobj)
        #print('mask_array.shape',array2.shape)
        #array_nb = np.array(array_nb.dataobj)
        channels = max_channal(array_nb)

        mask1 = choose_channal(maskimg, channels)

            
        #show_img(mask)
        #sel_channels = round((low_channels+up_channels)/ 2)
        print('sel_channels', channels)
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
        print('img_array.shape', img.shape)
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

        ROI1 = ROI_extraction_3(img, mask1, dataset.patient_ids[x])
        print('ROI1')
#        ROI2 = ROI_extraction(img, mask2, dataset.patient_ids[x])
#       print('ROI2')
#        ROI3 = ROI_extraction(img, mask3, dataset.patient_ids[x])
#        print('ROI3')
        #seeimage
        #ROI=tran_channal(ROI)

        print('ROI1.shape', ROI1.shape, dataset.search_string_img[x])
            
#       ROI1 = resize_max_img(ROI1)
#       seeimage(ROI1)
        print("此图像最大经线：",ROI1.shape)
#        print('ROI2.shape', ROI2.shape, dataset.search_string_img[x])
#       print('ROI3.shape', ROI3.shape, dataset.search_string_img[x])
#       ROI_TRANS = tran_RGB_XY1_RGB_XY3(ROI1)
#       ROI_TRANS=resize_128_128_16(ROI_TRANS)
        ROI1 = torch.from_numpy(ROI1).float()
        input_tensor_ROI1 = ROI1.permute(2, 0, 1).unsqueeze(0)
        
        print('input_tensor_ROI1',input_tensor_ROI1.shape)
        final1 = input_tensor_ROI1
#       final2 = input_tensor_ROI1
#        final3 = np.dstack((ROI3,))

        
        final_array2.append(final1)

 
#       final_array2.append(final2)
#       final_array2.append(final2)
#       print('final_array2shape',final_array2.shape)

        
    X_tot = final_array2
        
    print(X_tot[0].shape)
        
    return X_tot


train = get_dataset_long(data_dir)
input_tensor = torch.cat(train, dim=0)

y_train = get_classes(data_dir)

input_data = []

for image_np, label in zip(train , y_train):

    # Convert NumPy array to PyTorch tensor
#   input_tensor = torch.from_numpy(image_np).float()

    # Resize tensor to (1,3,224,224)
#   input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

    # Append tensor to the list of input tensors along with its label
    input_data.append((input_tensor, label))
    
    

train_loader = torch.utils.data.DataLoader(input_tensor, batch_size=512, shuffle=True)
from torchvision.utils import make_grid

for images in train_loader :
    print('Images Shape:', images.shape)
    plt.figure()
    plt.axis('off')
    plt.imshow(make_grid(images).permute(1, 2, 0))

torch.save(input_tensor, 'touch_2d_X1_input_tensor.pt')
torch.save(y_train, 'touch_2d_X1_y_train.pt')

