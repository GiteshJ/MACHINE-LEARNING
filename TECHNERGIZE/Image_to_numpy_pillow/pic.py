# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:06:39 2018

@author: Gitesh Jain
"""
#Variables
CORRECT_labels=[]
RGB_image=[]
GRAYSCALE_image=[]
RESIZED_image=[]
Types=["Benign","In Situ","Invasive","Normal"]
Number_of_images=[68,62,61,54]
y=index=-1
NPY_file=["rgb","grayscale","resized","correctlabels"]
#packages
from PIL import Image
from numpy import array
import numpy as np
import pathlib
for j in Types:
    y=y+1
    index=index+1
    for i in range(1,Number_of_images[index]):
        path=j+"/"
        n="t"+str(i)+".tif"
        fullpath= path+n
        im = Image.open(pathlib.Path(fullpath))  
        #RGB format
        arr = array(im)
        RGB_image.append(arr)
        #grayscale format
        im=im.convert('L')   
        arr = array(im)
        GRAYSCALE_image.append(arr)
        #Resized format 400*400 
        im=im.resize((400, 400))
        arr = array(im)
        RESIZED_image.append(arr)
        CORRECT_labels.append(y)
    print("DONE" + j)
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
