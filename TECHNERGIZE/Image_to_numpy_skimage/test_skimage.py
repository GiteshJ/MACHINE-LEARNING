# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:47:45 2018

@author: GITESH
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:50:04 2018

@author: Gitesh Jain
"""

CORRECT_labels=[]
RGB_image=[]
GRAYSCALE_image=[]
GRAYSCALE_resized_image=[]
RESIZED_image=[]
Types=["Test_data"]

index=36
NPY_file=["rgb_sk_test","grayscale_sk_test","grayscale_resized_sk_test","rgb_resized_sk_test","correctlabels_sk_test"]
#packages

from skimage import io,color
from skimage.transform import resize

from numpy import array
import numpy as np

for j in Types:
    
  
    for i in range(0,36):
        path=j+"/"
        n=str(i)+".tif"
        fullpath= path+n
        im = io.imread((fullpath))  
        #RGB format
        RGB_image.append(im)
        #grayscale format
        im1=color.rgb2grey(im)          
        GRAYSCALE_image.append(im1)
        #Resized Grayscale format
        im2=resize(im1,(400,400),mode='constant')
        GRAYSCALE_resized_image.append(im2)
        
        #RGB resized image
        im2=resize(im,(400, 400,3),mode='constant')
        RESIZED_image.append(im2)
        
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
#RGB_image=array(RGB_image)
#GRAYSCALE_image=array(GRAYSCALE_image)
GRAYSCALE_resized_image=array( GRAYSCALE_resized_image)
RESIZED_image=array(RESIZED_image)
CORRECT_labels=array(CORRECT_labels)
CORRECT_labels=np.reshape(CORRECT_labels,[CORRECT_labels.shape[0],1])
#Save as npy files
#np.save(NPY_file[0],RGB_image )
#np.save(NPY_file[1],GRAYSCALE_image)
np.save(NPY_file[2],GRAYSCALE_resized_image)
np.save(NPY_file[3],RESIZED_image)
np.save(NPY_file[4],CORRECT_labels)
#Verify shapes
#print(RGB_image.shape)    
#print(GRAYSCALE_image.shape)
print( GRAYSCALE_resized_image.shape)
print(RESIZED_image.shape)
print(CORRECT_labels.shape)
