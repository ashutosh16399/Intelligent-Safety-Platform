# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:02:49 2020

@author: Ashut
"""


import numpy as np
import os
import keras.backend
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from keras import backend as K
from sklearn.model_selection import train_test_split





#PATH = os.getcwd()
# Define data path
data_path ='C:/Users/Ashut/Desktop/projects/data/train'
data_dir_list = os.listdir(data_path)
print(data_dir_list)

img_data_list=[]
labels=[]

for dataset in data_dir_list:
  img_list=os.listdir(data_path+'/'+ dataset)
  print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
  for img in img_list:
    try: 
      img_path = data_path + '/' + dataset + '/' + img
      img = image.load_img(img_path,target_size=(224,224))
      x = image.img_to_array(img)
      x = np.expand_dims(x,axis=0)
      x = preprocess_input(x)
      labels.append(data_dir_list.index(dataset))
      print('Input image shape:',x.shape)
      print(data_dir_list.index(dataset))
      img_data_list.append(x)
    except OSError:
      pass
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
'''labels = np.ones((num_of_samples,),dtype='int64')

labels[0:2679]=0
labels[2679:12401]=1
labels[12401:19050]=2
labels[19050:21083]=3'''

names = ['drugs','normal','pornographic','unpleasant visuals']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
x_train = x[0:14760]
x_test = x[14760:21063]
y_train = y[0:14760]
y_test = y[14760:21063]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



#image_input = Input(shape=(224, 224, 3))



model = ResNet50(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(1024, activation='relu',name='fc-1')(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu', name = 'fc-2')(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', name = 'fc-3')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu',name='fc-4')(x)
x = Dropout(0.2)(x)
# a softmax layer for 4 classes
out = Dense(4, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-26]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.summary()

checkpoint = ModelCheckpoint(
                             'image_classificationDEEEEEEEPER32.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(
                          monitor='val_loss',
                          min_delta=0,
                          patience=4,
                          verbose=1,restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.0001)

callbacks = [earlystop,checkpoint,learning_rate_reduction]

custom_resnet_model2.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
              )




import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

'''from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True'''

#epochs = 25
#batch_size = 100

hist = custom_resnet_model2.fit(x_train, y_train, batch_size=30, epochs=10, verbose=1, validation_data=(x_test, y_test),callbacks=callbacks)
(loss, accuracy) = custom_resnet_model2.evaluate(x_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

'''TRAINING_DIR="data/train"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR="data/val"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(224,224),
	class_mode='categorical',
  batch_size=100
)

validation_generator=validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(224,224),
	class_mode='categorical',
  batch_size=100
)


try:
    history = custom_resnet_model2.fit_generator(train_generator, epochs=25, steps_per_epoch=10, validation_data = validation_generator, verbose = 1, validation_steps=10)
except IOError:
    pass # You can always log it to logger
    
    
nb_train_samples = 17502
nb_validation_samples = 3596  


history = custom_resnet_model2.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size)'''


