# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:25:12 2018

@author: GITESH
"""

from skimage import io,color
from skimage.transform import resize
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation,merge , ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from numpy import array
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

bn='bn_layer_'
conv='conv_layer_'
fc= 'fc_layer_'
k=32
def load_dataset():
    fullpath='ChestXray-NIHCC/images/00000001_000.png'
    im = io.imread((fullpath))  
    #plt.imshow(im)
    print(im.shape)
    im_resized=resize(im,(224,224,1),mode='constant')
    l=[]
    l.append(im_resized)
    X_train=array(l)
    l=[]
    Y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    l.append(Y)
    Y_train=array(l)
    fullpath='ChestXray-NIHCC/images/00000001_001.png'
    im = io.imread((fullpath))  
    #plt.imshow(im)
    print(im.shape)
    im_resized=resize(im,(224,224,1),mode='constant')
    l=[]
    l.append(im_resized)
    X_test=array(l)
    l=[]
    Y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    l.append(Y)
    Y_test=array(l)
    return X_train, Y_train,X_test,Y_test,14

def bottleneck_composite(l,layer):
    # bottleneck layer
    X=l
    if type(l) is list:
        if(len(l)==1):
            X=l[0]
        else:
            X= merge(l, mode='concat', concat_axis=-1)

    X = BatchNormalization(axis = 3, name = bn + str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(4*k, (1, 1), strides = (1, 1),padding='same', name = conv + str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    # Composite layer
    layer=layer+1
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (3, 3), strides = (1, 1),padding='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    return X
    
    
layer=0    
def xnet(classes=14,input_shape=(224,224,1),):
    X_input = Input(input_shape)
    layer=0
    layer=layer+1
    X = ZeroPadding2D((3, 3))(X_input)
    X = BatchNormalization(axis = 3, name = bn + str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(2*k, (7, 7), strides = (2, 2), name = conv + str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    #Dense Block = 1
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,5):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)

    # Transition layer = 1   
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1, 1))(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    
    
    #Dense Block = 2
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,11):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    
    
    # Transition layer = 2
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1, 1))(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    
    #Dense Block = 3
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,23):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    
    # Transition layer = 3
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = ZeroPadding2D((1, 1))(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    
    #Dense Block = 4
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,15):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    layer=layer+2
    X=  GlobalAveragePooling2D()(X)
    # fully connected layer
    #X = Flatten()(X)
    X = Dense(classes, activation='softmax', name=  fc  +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X, name="DenseNet121")
    
    return model


model = xnet(classes = 14,input_shape = (224,224,1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.




print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model.fit(X_train, Y_train, epochs = 2, batch_size = 1)  
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))   
