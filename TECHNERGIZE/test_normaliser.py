#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:54:05 2018

@author: praveenbaheti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:04:15 2018

@author: praveenbaheti
"""
from skimage import io,color
import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane
import numpy as np
import matplotlib.pyplot as plt


#from skimage import io,color
#from skimage.transform import resize

#from numpy import array
norm=stainNorm_Macenko.normalizer()
#n.fit(utils.read_image('Benign/t0.tif'))

import numpy as np

print("BENIGN")
for i in range(1,10):
    path="Benign"+"/"
    path_norm="Benign_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        #print(i1)
        
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)

print("IN SITU")        
for i in range(1,10):
    path="In Situ"+"/"
    path_norm="In Situ_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        #print(i1)
        
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)
        
print("INVASIVE")
for i in range(1,10):
    path="Invasive"+"/"
    path_norm="Invasive_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        #print(i1)
        
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)
print("NORMAL")
for i in range(1,10):
    path="Normal"+"/"
    path_norm="Normal_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        #print(i1)
    
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)