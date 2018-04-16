from skimage import io,color
from skimage.transform import resize
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense,Dropout,Concatenate, Activation,merge , ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.initializers import he_uniform
import scipy.misc
from keras.optimizers import *
from matplotlib import pyplot as plt
from numpy import array
from matplotlib.pyplot import imshow,savefig
import keras.backend as K
from keras.applications import densenet
from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
import json
import tensorflow as tf
import sklearn
from sklearn.metrics import precision_recall_fscore_support,classification_report,confusion_matrix,roc_auc_score,confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
K.get_session().run(tf.local_variables_initializer())
K.get_session().run(tf.global_variables_initializer())
#using sklelarn metrics
def class_report_metric(y_true,y_pred):
    yt=np.array(y_true)
    yp=np.array(K.round(y_pred))
    st=classification_report(yt, yp)
    return st
def auroc(y_true,y_pred):
    yt=np.array(y_true)
    yp=np.array((y_pred))
    st=roc_auc_score(yt, yp)
    return st
#using Tensorflow metrics
def auroc_metric(y_true,y_pred):

    yt=y_true
    yp=y_pred
    auroc=tf.metrics.auc(yt, yp)[0]
    K.get_session().run(tf.local_variables_initializer())
    return auroc
def recall_metric(y_true, y_pred):
    yt=y_true
    yp=K.round(y_pred)
    recall=tf.metrics.recall(yt, yp)[0]
    return recall
def precision_metric(y_true, y_pred):
    yt=y_true
    yp=K.round(y_pred)
    precision=tf.metrics.recall(yt, yp)[0]

    return precision
def f1_metric(y_true, y_pred):
    precision=precision_metric(y_true, y_pred)[0]
    recall=recall_metric(y_true, y_pred)[0]
    f1=(2*precision*recall)/(precision+recall)
    return f1

adam=Adam(lr=0.001)
def cnnmodel(classes=14,input_shape=(224,224,3)):
    X_input = Input(input_shape)
    model1=densenet.DenseNet121(include_top=False, weights='imagenet')
    X = model1.layers[-1].output
    print(X.shape)
    X=  GlobalAveragePooling2D()(X)
    print(X.shape)
    X=Dense(classes, activation='sigmoid', name=  'fc121', kernel_initializer = he_uniform(seed=0))(X)
    print(X.shape)
    model = Model(model1.input,outputs=X,name="CNN")
    return model
model = cnnmodel(classes = 1,input_shape = (224,224,3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[class_report_metric])
#model.summary()

train_datagen = ImageDataGenerator( rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Train',
        target_size=(224,224),
        batch_size=64,
        class_mode='binary')
validation_generator = val_datagen.flow_from_directory(
        'Val',
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary')
test_generator=test_datagen.flow_from_directory(
        'Test',
        target_size=(224,224),
        batch_size=64,
        class_mode='binary')
print(train_generator.class_indices)
print(test_generator.class_indices)
print(validation_generator.class_indices)
history=model.fit_generator(train_generator, epochs =10,steps_per_epoch=50,validation_data=validation_generator, validation_steps=50)
model_files='mymodel'
model.save(model_files)
print(history.history)
model=load_model(total_model[-1], custom_objects={'class_report_metric': class_report_metric})
preds = model.evaluate_generator(train_generator, steps=50)
print(preds)
preds = model.evaluate_generator(validation_generator, steps=50)
print(preds)
preds = model.evaluate_generator(test_generator, steps=50)
print(preds)
print("DONE")
