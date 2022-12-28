
import numpy as np
import os
import pprint as pp
import pandas as pd
import glob
import pdb
import nibabel as nib
import radiomics as rm
import pprint as pp
import matplotlib.pyplot as plt
import SimpleITK as sitk
#get_ipython().run_line_magic('matplotlib', 'inline')
import csv
import re

#pos_path= r/home/zy/Downloads/xn/training/pos'
test_path = r'C:/Users/zy/Desktop/testing'
#all_path= r'/home/zy/Downloads/xn/testing'
'''
for file in os.walk(pos_path):
    if file[0] == pos_path:
        continue
    print('file[0]',file[0])
    search_string = os.path.join(pos_path,  os.path.basename(file[0]) + ".nii.gz")
    #print(file[1])
    print('os.path.basename(file[0])',os.path.basename(file[0]))
    print('search_string',search_string)
    print('file[0]',file[0],'file[2][0]',file[2][0],'file[2][1]',file[2][1])
    print("\n")
'''

def selname(file,target_file_name):
    
    for a in range(len(file[2])):
        datanames = file[2][a]
        #print("datanames",datanames)
        pattern = re.compile(target_file_name)
        match = pattern.search(datanames)
        if match !=None:
            print('return',file[2][a]) 
            return file[2][a]
        
        
            
        
    
    

class Dataset():

    def __init__(self, path, counter_max=0, type='4D'):

        self.path = path
        self.class_folders = [folder for folder in os.listdir(self.path) if 'class' in folder]
        print('self.class_folders',self.class_folders)
        #self.dataset = {'img_filenames': [], 'frame_ed': [], 'msk_ed': [], 'frame_es':[], 'msk_es':[]}
        self.dataset = {'img': [], 'mask': []}
        # self.es_frames = pd.read_excel(os.path.join(self.path,"ES_frames.xlsx"))
        self.patient_ids = []

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
            '''
            self.dataset['frame_ed'].append(os.path.join(file[0], file[2][2]))  #mask
            self.dataset['msk_ed'].append(os.path.join(file[0], file[2][3]))  #image #dont pay attention to the "index"
            self.dataset['frame_es'].append(os.path.join(file[0], file[2][-2]))  #mask
            self.dataset['msk_es'].append(os.path.join(file[0], file[2][-1])) 
            '''
            #self.dataset['img'].append(os.path.join(file[0], file[2][1]))
            self.dataset['img'].append(os.path.join(file[0], selname(file,'img')))#mask
            self.dataset['mask'].append(os.path.join(file[0], selname(file,'mask')))  #image #dont pay attention to the "index"
            #self.dataset['frame_es'].append(os.path.join(file[0], file[2][-2]))  #mask
            #self.dataset['msk_es'].append(os.path.join(file[0], file[2][-1])) 
            patient_id = os.path.basename(file[0])
            #search_string = os.path.join(path,  patient_id , patient_id + file[2][1] + ".nii.gz")  #what is this for really?

            #     pdb.set_trace()
            #image_location = glob.glob(search_string)

            self.patient_ids.append(patient_id)

            #self.dataset['img'].append(image_location)

            #print(self.dataset['img'][-1])
            #print(self.dataset['maks'][-1])
            #print(self.dataset['frame_es'][-1])
            #print(self.dataset['msk_es'][-1])
            #print('img_filenames')
            print('img',self.dataset['img'][-1])
            print('mask',self.dataset['mask'][-1])
            #print(self.dataset['frame_es'][-1])
            #print(self.dataset['msk_es'][-1])
            print('/n')




test_dataset = Dataset(test_path)



len(test_dataset.patient_ids)

print(test_dataset.patient_ids)
print('len',len(test_dataset.patient_ids))
i = 0
feature_names = []
feature_values = []

print('step 1 complete####################################################################')


for x in range(len(test_dataset.patient_ids)):
    print(" \n Patient ID: ", x, "\n")
    #if x == 1:
        #break
    img = nib.load(test_dataset.dataset['img'][x]).get_fdata()
    msk = nib.load(test_dataset.dataset['mask'][x]).get_fdata()
    print('img',test_dataset.dataset['img'][x],'mask',test_dataset.dataset['mask'][x])
    #print('img_ED',img_ED,'msk_ED',msk_ED)
    #img_ES = nib.load(dataset.dataset['frame_es'][x]).get_fdata()
    #msk_ES = nib.load(dataset.dataset['msk_es'][x]).get_fdata()
    
    
    sitk_img = sitk.GetImageFromArray(np.array(img, dtype=np.int16))   
    sitk_msk = sitk.GetImageFromArray(np.array(msk, dtype=np.int16))
    
    #sitk_img_ES = sitk.GetImageFromArray(np.array(img_ES, dtype=np.int16))   
    #sitk_msk_ES = sitk.GetImageFromArray(np.array(msk_ES, dtype=np.int16))
    
    #print('sitk_img_ES',sitk_img_ES,'sitk_msk_ES',sitk_msk_ES)
    
    #create an radiomics extractor

    extractor = rm.featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['label'] =  1
    #features = extractor.execute(filpath_img, filpath_msk) # from file
    features = extractor.execute(sitk_img, sitk_msk) # from sitk image
    
    #feature_names = []
    #feature_values = []
    all_subjects = []

    for key,value in features.items():
    
#         if 'shape' in key:
        print('label1',key,value) 
        #print('value',)
        feature_names.append(key+'_test')   ## fen lei qi
        feature_values.append(value)
    # LV EXTRACTION : LABEL 1
    print(len(feature_values)) 
    df = pd.DataFrame([feature_values], columns = feature_names)

            
'''   
    extractor.settings['label'] = 3   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    
    # RV EXTRACTION : LABEL 3
    for key,value in features.items():
    
#         if 'shape' in key:
        print('label3',key,value) 
        feature_names.append(key+'_RV_ED')
        feature_values.append(value)
    
    print(len(feature_values))
          
    extractor.settings['label'] = 2   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    # MYOCARDIUM EXTRACTION : LABEL 2
    for key,value in features.items():
    
#         if 'general' not in key:
        print('label2',key,value) 
        feature_names.append(key+'_MYO_ED')
        feature_values.append(value)
            
    print(len(feature_values))
'''
    

print('step 2 complete####################################################################')





list_test= []
count = 0
len(df.columns)
for i in df.columns:
    if i == "diagnostics_Versions_PyRadiomics_test":
        list_test.append(df.iloc[0,count:count+129])
        count = count + (129)

print('Number of Patients Loaded: ', len(list_test))

df_2 = pd.DataFrame(list_test)

### Extracting End Systole

'''
i = 0
feature_names = []
feature_values = []

neg_dataset = Dataset(neg_path)


for x in range(len(neg_dataset.patient_ids)):
    print(" \n Patient ID: ", x, "\n")
    #if x == 1:
        #break
    img = nib.load(neg_dataset.dataset['img'][x]).get_fdata()
    msk = nib.load(neg_dataset.dataset['mask'][x]).get_fdata()
    print('img',(neg_dataset.dataset['img'][x],'mask',(neg_dataset.dataset['mask'][x])))
    #print('img_ED',img_ED,'msk_ED',msk_ED)
    #img_ES = nib.load(dataset.dataset['frame_es'][x]).get_fdata()
    #msk_ES = nib.load(dataset.dataset['msk_es'][x]).get_fdata()
    
    
    sitk_img = sitk.GetImageFromArray(np.array(img, dtype=np.int16))   
    sitk_msk = sitk.GetImageFromArray(np.array(msk, dtype=np.int16))
    
    #sitk_img_ES = sitk.GetImageFromArray(np.array(img_ES, dtype=np.int16))   
    #sitk_msk_ES = sitk.GetImageFromArray(np.array(msk_ES, dtype=np.int16))
    
    #print('sitk_img_ES',sitk_img_ES,'sitk_msk_ES',sitk_msk_ES)
    
    #create an radiomics extractor

    extractor = rm.featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['label'] =  1
    #features = extractor.execute(filpath_img, filpath_msk) # from file
    features = extractor.execute(sitk_img, sitk_msk) # from sitk image
    
    #feature_names = []
    #feature_values = []
    all_subjects = []

    for key,value in features.items():
    
#         if 'shape' in key:
        print('label1',key,value) 
        #print('value',)
        feature_names.append(key+'_negitive')   ## fen lei qi
        feature_values.append(value)
    # LV EXTRACTION : LABEL 1
    print(len(feature_values)) 
    df = pd.DataFrame([feature_values], columns = feature_names)

'''            
'''   
    extractor.settings['label'] = 3   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    
    # RV EXTRACTION : LABEL 3
    for key,value in features.items():
    
#         if 'shape' in key:
        print('label3',key,value) 
        feature_names.append(key+'_RV_ED')
        feature_values.append(value)
    
    print(len(feature_values))
          
    extractor.settings['label'] = 2   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    # MYOCARDIUM EXTRACTION : LABEL 2
    for key,value in features.items():
    
#         if 'general' not in key:
        print('label2',key,value) 
        feature_names.append(key+'_MYO_ED')
        feature_values.append(value)
            
    print(len(feature_values))
'''
    

print('step 3 complete####################################################################')


'''
list_negitive= []
count = 0
for i in df.columns:
    if i == "diagnostics_Versions_PyRadiomics_negitive":
        list_negitive.append(df.iloc[0,count:count+129])     ##feature==129
        count = count + (129)


# In[19]:


print('Number of Patients Loaded', len(list_negitive))


df_3 = pd.DataFrame(list_negitive)



### Extracting End Systole


i = 0
feature_names = []
feature_values = []

all_dataset = Dataset(all_path)


for x in range(len(all_dataset.patient_ids)):
    print(" \n Patient ID: ", x, "\n")
    #if x == 1:
        #break
    img = nib.load(all_dataset.dataset['img'][x]).get_fdata()
    msk = nib.load(all_dataset.dataset['mask'][x]).get_fdata()
    print('img',(all_dataset.dataset['img'][x],'mask',(all_dataset.dataset['mask'][x])))
    #print('img_ED',img_ED,'msk_ED',msk_ED)
    #img_ES = nib.load(dataset.dataset['frame_es'][x]).get_fdata()
    #msk_ES = nib.load(dataset.dataset['msk_es'][x]).get_fdata()
    
    
    sitk_img = sitk.GetImageFromArray(np.array(img, dtype=np.int16))   
    sitk_msk = sitk.GetImageFromArray(np.array(msk, dtype=np.int16))
    
    #sitk_img_ES = sitk.GetImageFromArray(np.array(img_ES, dtype=np.int16))   
    #sitk_msk_ES = sitk.GetImageFromArray(np.array(msk_ES, dtype=np.int16))
    
    #print('sitk_img_ES',sitk_img_ES,'sitk_msk_ES',sitk_msk_ES)
    
    #create an radiomics extractor

    extractor = rm.featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['label'] =  1
    #features = extractor.execute(filpath_img, filpath_msk) # from file
    features = extractor.execute(sitk_img, sitk_msk) # from sitk image
    
    #feature_names = []
    #feature_values = []
    all_subjects = []

    for key,value in features.items():
    
#         if 'shape' in key:
        print('label1',key,value) 
        #print('value',)
        feature_names.append(key+'_all')   ## fen lei qi
        feature_values.append(value)
    # LV EXTRACTION : LABEL 1
    print(len(feature_values)) 
    df = pd.DataFrame([feature_values], columns = feature_names)

'''            
'''   
    extractor.settings['label'] = 3   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    
    # RV EXTRACTION : LABEL 3
    for key,value in features.items():
    
#         if 'shape' in key:
        print('label3',key,value) 
        feature_names.append(key+'_RV_ED')
        feature_values.append(value)
    
    print(len(feature_values))
          
    extractor.settings['label'] = 2   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    # MYOCARDIUM EXTRACTION : LABEL 2
    for key,value in features.items():
    
#         if 'general' not in key:
        print('label2',key,value) 
        feature_names.append(key+'_MYO_ED')
        feature_values.append(value)
            
    print(len(feature_values))
'''
    
'''
print('step 3 complete####################################################################')



list_all= []
count = 0
for i in df.columns:
    if i == "diagnostics_Versions_PyRadiomics_all":
        list_all.append(df.iloc[0,count:count+129])     ##feature==129
        count = count + (129)


# In[19]:


print('Number of Patients Loaded', len(list_all))


df_4 = pd.DataFrame(list_all)




#df_radiomcis = pd.concat((df_2, df_3), axis=1)
'''
df_radiomcis=df_2


df_radiomcis.to_csv('Testing.csv')





mylines = []
#height_list = []
#weight_list = []
for file in os.walk(test_path):
    cont = 0
    if file[0] == test_path:
        continue
    print(os.path.join(file[0],file[2][0]))
    #search_string = os.path.join(seg_path, os.path.basename(file[0]),  os.path.basename(file[0]) + ".nii.gz")
    #print(file[1])
    #print(os.path.basename(file[0]))
#     #print(search_string)
#     print(file[2][0])
#     print(file[2][4],  "\n")
    

    with open ((os.path.join(file[0], selname(file,'Info.cfg')))) as myfile:
        for myline in myfile:
            #cont += 1
            #print('myline',myline,'cont',cont)

            #if cont==3:
            classs = myline.lstrip("Group: ")
#           classs = classs.rstrip("\n")
            mylines.append(classs)
'''
            if cont == 4:
                height = myline.lstrip("Height: ")
                height = height.rstrip("\n")
                height_list.append(height)
            if cont == 6:
                weight = myline.lstrip("Weight: ")
                weight = weight.rstrip("\n")
                weight_list.append(weight)
            #print(cont)
'''

class_df = pd.DataFrame(mylines)
#height_df = pd.DataFrame(height_list)
#weight_df = pd.DataFrame(weight_list)



#df_radiomcis['height']= height_list
#df_radiomcis['weight']= weight_list
df_radiomcis['class']= mylines



df_radiomcis.head()

df_radiomcis.to_csv('(Radiomics+Clinical)_testing.csv')

