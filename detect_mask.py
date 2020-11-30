from model import *
from model.MobileNetV2 import *
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout

mob = MobileNetV2(
    input_shape = (128,128,3),
    include_top = False,
    weights = 'imagenet',
    )
mob.trainable = False

def classifier_MobileNetV2(n_classes = 2):

    model = Sequential()
    model.add(mob)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes,activation='softmax'))

    return model


def load_model(weights_dir = 'weights/test.h5',n_classes = 2):
    model = classifier_MobileNetV2(n_classes);
    model.load_weights(weights_dir);
    return model
    

def detect_mask(model,img ):

	try: 

		img_resized = cv2.resize(img, (128, 128) ,interpolation = cv2.INTER_CUBIC)
		img_resized = (img_resized[np.newaxis,:]/255.).astype('float32')
		prediction =  model.predict(img_resized)
		return [prediction[0,0] , prediction[0,1]]

	except Exception as e:
		print('Mask detection: exception ! ')
		return [-1, -1]


def detect_mask_3_classes(model,img):

    try: 

        img_resized = cv2.resize(img, (128, 128) ,interpolation = cv2.INTER_CUBIC)
        img_resized = (img_resized[np.newaxis,:]/255.).astype('float32')
        prediction =  model.predict(img_resized)
        return [prediction[0,0] , prediction[0,1] ,prediction[0,2] ]

    except Exception as e:
        print('Mask detection: exception ! ')
        return [-1, -1, -1]