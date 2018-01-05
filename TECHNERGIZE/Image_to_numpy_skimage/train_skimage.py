# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:23:57 2018

@author: GITESH
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:06:39 2018

@author: Gitesh Jain
"""
#Variables
CORRECT_labels=[]
RGB_image=[]
GRAYSCALE_image=[]
GRAYSCALE_resized_image=[]
RESIZED_image=[]
Types=["Benign","In Situ","Invasive","Normal"]
Number_of_images=[68,62,61,54]
y=index=-1
NPY_file=["rgb_train","grayscale_train","grayscale_resized_train","rgb_resized_train","correctlabels_train"]
#packages
from skimage import io,color
from skimage.transform import resize
from numpy import array
import numpy as np

for j in Types:
    y=y+1
    index=index+1
    for i in range(1,Number_of_images[index]):
        path=j+"/"
        n="t"+str(i)+".tif"
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
        CORRECT_labels.append(y)
    print("DONE" + j)
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
