import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, GlobalAveragePooling2D

mob = MobileNetV2(
	input_shape = (128,128,3),
	include_top = False,
	weights = 'imagenet',
	)
mob.trainable = False

def classifier_MobileNetV2():

	model = Sequential()
	model.add(mob)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2,activation='softmax'))

	return model

