from model import *
from model.MobileNetV2 import *
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout


def load_model(weights_dir = 'model.h5'):

    mob = MobileNetV2(
        input_shape = (128,128,3),
        include_top = False,
        weights = 'imagenet',
        classes=2,
        classifier_activation="softmax",
    )
    mob.trainable = True
    
    
    # In[15]:
    
    
    model = Sequential()
    model.add(mob)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(2,activation='softmax'))
    model.summary()
    model.load_weights(weights_dir)


def detect_mask(model,img):

	try: 

		img_resized = cv2.resize(img, (128, 128) ,interpolation = cv2.INTER_CUBIC)
		img_resized = (img_resized[np.newaxis,:]/255.).astype('float32')
		prediction =  model.predict(img_resized)
		return [prediction[0,0] , prediction[0,1]]

	except Exception as e:
		print('Mask detection: exception ! ')
		return [-1, -1]