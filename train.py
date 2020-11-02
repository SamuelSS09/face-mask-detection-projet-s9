import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
import shutil

from shutil import copyfile

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

base = ''

DATAPATH = os.path.join(base,'datasets/our_dataset/')
MASKPATH = os.path.join(base,'datasets/our_dataset/mask/')
NOMASKPATH = os.path.join(base,'datasets/our_dataset/no_mask/')
TESTPATH = os.path.join(base,'datasets/our_dataset/testdata/')
TRAINPATH = os.path.join(base,'datasets/our_dataset/traindata/')

WEIGHTS_PATH = os.path.join(base,'weights/')
os.makedirs(WEIGHTS_PATH,exist_ok = True)
WEIGHTS_FILE = os.path.join(WEIGHTS_PATH,'weights_v2.h5')

BATCH_SIZE = 8
EPOCHS = 15
STEPS_PER_EPOCH = np.floor((2188 + 2400) / BATCH_SIZE ).astype('uint8')

print(STEPS_PER_EPOCH)

TRAIN_MASK_PATH = os.path.join(base,'datasets/our_dataset/traindata/mask/')
TRAIN_NOMASK_PATH = os.path.join(base,'datasets/our_dataset/traindata/no_mask/')
TEST_MASK_PATH = os.path.join(base,'datasets/our_dataset/testdata/mask/')
TEST_NOMASK_PATH = os.path.join(base,'datasets/our_dataset/testdata/no_mask/')

os.makedirs(TRAIN_MASK_PATH,exist_ok = True)
os.makedirs(TRAIN_NOMASK_PATH,exist_ok = True)
os.makedirs(TEST_MASK_PATH,exist_ok = True)
os.makedirs(TEST_NOMASK_PATH,exist_ok = True)

trainGen = ImageDataGenerator(rescale=1.0/255.,
                              rotation_range=40,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              fill_mode='nearest')

testGen = ImageDataGenerator(
    rescale= 1.0/255.,
)

train = trainGen.flow_from_directory(
    TRAINPATH, 
    target_size=(128, 128),
    classes=['mask','no_mask'],
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    subset='training'
)

validation = testGen.flow_from_directory(
    TESTPATH, 
    target_size=(128, 128),
    classes=['mask','no_mask'],
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    shuffle=False,
)

a = next(validation)

# i  = np.floor(255*a[0][0,:]).astype('uint8')

# cv2.imshow('im1 ', i) 
  
# # waits for user to press any key 
# # (this is necessary to avoid Python kernel form crashing) 
# cv2.waitKey(0) 
  
# # closing all open windows 
# cv2.destroyAllWindows()

# i  = np.floor(255*a[0][1,:]).astype('uint8')

# cv2.imshow('im2 ', i) 
  
# # waits for user to press any key 
# # (this is necessary to avoid Python kernel form crashing) 
# cv2.waitKey(0) 
  
# # closing all open windows 
# cv2.destroyAllWindows()


# mob = MobileNetV2(
#     input_shape = (128,128,3),
#     include_top = False,
#     weights = 'imagenet',
# )
# mob.trainable = False

# model = Sequential()
# model.add(mob)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2,activation='softmax'))
# model.summary()


# In[16]:


# model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])


# In[17]:


# checkpoint = ModelCheckpoint(
#     'model.h5',
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True,
#     save_weights_only=True,
#     mode='min'
# )


# In[ ]:


# history = model.fit(
#     train,
#     epochs = EPOCHS,
#     steps_per_epoch = STEPS_PER_EPOCH,
#     validation_data = validation,
#     callbacks = [checkpoint]
# )

# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.grid()
# plt.savefig('')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.grid()
# plt.show()