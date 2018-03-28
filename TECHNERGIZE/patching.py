# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:39:39 2018

@author: GITESH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:08:06 2018

@author: GITESH
"""
import warnings
from random import shuffle
#from sklearn.utils import shuffle
import numpy as np
from keras.utils import np_utils
import keras.backend as K
K.set_image_data_format('channels_last')
###
from skimage.transform import rotate

from skimage import io
from skimage.transform import resize
warnings.filterwarnings('ignore')
def save_insitu(image,count,fullpath):
    io.imsave(fullpath,image)
    
def save_benign(image,count,fullpath):
    #fullpath="Benign_test"+"/"+str(count)+".png"
    io.imsave(fullpath,image)
    #print(fullpath)
def save_invasive(image,count,fullpath):
    io.imsave(fullpath,image)
    #print(fullpath)
def save_normal(image,count,fullpath):
    io.imsave(fullpath,image)
    #print(fullpath)    
def read_insitu():
    patches=[]
    #y=1
    l=[]
    for i in range (0,72):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)
    count=0
    for i in range(0,72):
        num=l[i]
        image="In Situ/" +"t"+str(num)+".tif"
        
        #print(fullpath)
        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    
                    if(i<5):
                        fullpath="Test/In Situ/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/In Situ/"
                    else:
                        fullpath="Train/In Situ/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)
        #return RESIZED_image,CORRECT_labels
def read_Benign():
    patches=[]
    #y=1
    l=[]
    for i in range (0,78):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)

    count=0
    for i in range(0,78):
        num=l[i]
        image="Benign/" +"t"+str(num)+".tif"

        
        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    if(i<5):
                        fullpath="Test/Benign/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/Benign/"
                    else:
                        fullpath="Train/Benign/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)
        #return RESIZED_image,CORRECT_labels
def read_invasive():
    patches=[]
    #y=1
    l=[]
    for i in range (0,71):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)

    count=0
    for i in range(0,71):
        num=l[i]
        image="Invasive/" +"t"+str(num)+".tif"

        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    if(i<5):
                        fullpath="Test/Invasive/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/Invasive/"
                    else:
                        fullpath="Train/Invasive/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)
        #return RESIZED_image,CORRECT_labels
def read_normal():
    patches=[]
    #y=1
    l=[]
    for i in range (0,65):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)

    count=0
    for i in range(0,65):
        num=l[i]
        image="Normal/" +"t"+str(num)+".tif"
        
        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    if(i<5):
                        fullpath="Test/Normal/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/Normal/"
                    else:
                        fullpath="Train/Normal/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)

#read_insitu()
#read_Benign()
#read_invasive()          
read_normal()