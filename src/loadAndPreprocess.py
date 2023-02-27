import os,shutil
import keras.utils as image
import matplotlib.pyplot as plt
import numpy as np
import random

class LoadAndPreProcess:
#path to datasets
    
    def __init__(self,yes_path,no_path,base_dir):
        self.dataset_dir_no = no_path
        self.dataset_dir_yes = yes_path
        self.train_dataset_folder = os.path.join(base_dir,'train')
        self.test_dataset_folder = os.path.join(base_dir,'test')
        self.NOF_train = 150
        self.IMAGE_SIZE = 180
        self.yes_dataset = np.empty((0,self.IMAGE_SIZE,self.IMAGE_SIZE,3))
        self.no_dataset = np.empty((0,self.IMAGE_SIZE,self.IMAGE_SIZE,3))
        self.dataset = np.empty((0,self.IMAGE_SIZE,self.IMAGE_SIZE,3))
        self.label = label  = np.empty((0,1))

    def getNormalizeData(self):
        return self.dataset[:self.NOF_train].astype("float32")/255, self.dataset[self.NOF_train:].astype("float32")/255 ,self.label[:self.NOF_train],self.label[self.NOF_train:]

    def categorize(self):
        c1_idx = 0
        c2_idx = 0
        for indx in range(len(self.yes_dataset)+len(self.no_dataset)):
            classifier = np.array([random.randint(0, 1)],dtype=int)

            if (classifier==0 and c1_idx<len(self.no_dataset)):
                self.dataset = np.vstack((self.dataset,(self.no_dataset[c1_idx]).reshape(1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)))
                self.label = np.vstack((self.label, np.array([0])))
                c1_idx = c1_idx +1
            if (classifier ==1 and c2_idx<len(self.yes_dataset)):
                self.dataset = np.vstack((self.dataset,(self.yes_dataset[c2_idx]).reshape(1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)))
                self.label = np.vstack((self.label, np.array([1])))
                c2_idx = c2_idx +1


    def CreateFolderDataset(self):
        #os.mkdir(self.train_dataset_folder)
        #os.mkdir(self.test_dataset_folder)
        fnames_no = [os.path.join( self.dataset_dir_no, fname) for fname in os.listdir( self.dataset_dir_no)]
        fnames_yes = [os.path.join( self.dataset_dir_yes, fname) for fname in os.listdir( self.dataset_dir_yes)]
        img = image.load_img(fnames_no[3])
        #img = image.load_img(fnames_no[3],target_size=(150, 150))
        x = image.img_to_array(img)
        #plt.imshow(image.array_to_img(x)) 
        #plt.imshow(img) 
        
        for yes_img in fnames_yes:
            #x=image.img_to_array(yes_img)
            x = image.img_to_array(image.load_img(yes_img,target_size=[self.IMAGE_SIZE,self.IMAGE_SIZE]))
            self.yes_dataset = np.vstack((self.yes_dataset,x.reshape(1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)))

        #print(self.yes_dataset.shape)    
        for no_img in fnames_no:
            x = image.img_to_array(image.load_img(no_img,target_size=[self.IMAGE_SIZE,self.IMAGE_SIZE]))
            self.no_dataset = np.vstack((self.no_dataset,x.reshape(1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)))
        
        #print(self.no_dataset.shape)  
        self.categorize()  
        return self.dataset[:self.NOF_train], self.dataset[self.NOF_train:] ,self.label[:self.NOF_train],self.label[self.NOF_train:]