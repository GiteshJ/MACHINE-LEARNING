# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:17:03 2018

@author: gj
"""

import warnings
import numpy as np
from skimage import io
from skimage.transform import resize
from keras.models import load_model
import keras.backend as K

from skimage import io,color
import stain_utils as utils
import stainNorm_Macenko

def predictcancer(fullpath):
    fullpath = fullpath
    K.set_image_data_format('channels_last')
    warnings.filterwarnings('ignore')
    phase=""
    norm=stainNorm_Macenko.normalizer()
    fullpath_norm="outnorm.png"
    i1=utils.read_image(fullpath)
    norm.fit(i1)
    io.imsave((fullpath_norm),i1)


    im = io.imread((fullpath))
    im2=resize(im,(224,224,3),mode='constant')
    im2.resize((1,224,224,3))
    model=load_model('my_densenet')
    pred=model.predict(im2, batch_size=1, verbose=0, steps=None)
    #print((pred))
    category=np.argmax(pred,axis=-1)
    if(category==0):
        phase="Benign"
    elif(category==1):
        phase="In Situ"
    elif(category==2):
        phase="Invasive"
    elif(category==3):
        phase="Normal"

    chance=pred[0][category]*100
    return("PHASE = " + phase + "  " + "CHANCE(%) = "+ "%.2f" % chance)


















