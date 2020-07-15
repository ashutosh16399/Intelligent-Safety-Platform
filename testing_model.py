# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:29:57 2020

@author: Ashut
"""

from keras.models import load_model
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from keras import backend as K
import os
from keras.preprocessing import image
import numpy as np
from PIL import Image

names = ['drugs','normal','pornographic','unpleasant visuals']
model=load_model('image_classificationDEEPER32.h5')

data_path = "C:/Users/Ashut/Desktop/projects/test"
img_list=os.listdir(data_path)
for img in img_list:
    print (' images name-'+'{}\n'.format(img))
    img_path = data_path + '/' + img
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predic = model.predict(x)
    value=np.argmax(predic)
    print('predictions is- {}\n'.format(names[value]))
    if value == 2:
        print("image under review warning")
    elif value == 1:
        im = Image.open(img_path)
        im.show()
    else :
        print("see with caution may contain violent or addictive content")
        im = Image.open(img_path)
        im.show()
    
