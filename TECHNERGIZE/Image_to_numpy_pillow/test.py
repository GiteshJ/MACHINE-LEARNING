# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:50:04 2018

@author: Gitesh Jain
"""

CORRECT_labels=[]
RGB_image=[]
GRAYSCALE_image=[]
RESIZED_image=[]
Types=["E:/ML/Training_data/Test_data/"]
Number_of_images=[68,62,61,54]
index=36
NPY_file=["rgb_test","grayscale_test","resized_test","correctlabels_test"]
#packages
import os
from PIL import Image
from numpy import array
import numpy as np
import pathlib
for j in Types:
    
  
    for i in range(0,index):
        path=j+"/"
        n=str(i)+".tif"
        fullpath= path+n
        im = Image.open(pathlib.Path(fullpath))  
        #RGB format
        arr = array(im)
        RGB_image.append(arr)
        #grayscale format
        im1=im.convert('L')   
        arr = array(im1)
        GRAYSCALE_image.append(arr)
        #Resized format 400*400 
        im2=im.resize((400, 400,3))
        arr = array(im2)
        RESIZED_image.append(arr)
        
    print("DONE" + j)
#Reading the text labels from label.txt for test data
filepath='Test_data/labels.txt'
with open(filepath) as fp:  
   line = fp.readline()
   while line:
       words=line.split()
       if(words[1]=="Benign"):
           CORRECT_labels.append(0)
       elif(words[1]=="In"):
           CORRECT_labels.append(1)
       elif(words[1]=="Invasive"):
           CORRECT_labels.append(2)
       elif(words[1]=="Normal"):   
           CORRECT_labels.append(3)
       line = fp.readline()       
#Conversion to numpy arrays
RGB_image=array(RGB_image)
GRAYSCALE_image=array(GRAYSCALE_image)
RESIZED_image=array(RESIZED_image)
CORRECT_labels=array(CORRECT_labels)
CORRECT_labels=np.reshape(CORRECT_labels,[CORRECT_labels.shape[0],1])
#Save as npy files
np.save(NPY_file[0],RGB_image )
np.save(NPY_file[1],GRAYSCALE_image)
np.save(NPY_file[2],RESIZED_image)
np.save(NPY_file[3],CORRECT_labels)
#Verify shapes
print(RGB_image.shape)    
print(GRAYSCALE_image.shape)
print(RESIZED_image.shape)
print(CORRECT_labels.shape)
