from model import *
from model.MobileNetV2 import *
import cv2
import numpy as np


# from timeit import default_timer as timer




def load_model(weights_dir = 'weights/test.h5'):
	model = classifier_MobileNetV2()
	model.load_weights(weights_dir)
	return model

def detect_mask(model,img):

	try: 
		#start = timer()
		img_resized = cv2.resize(img, (128, 128) ,interpolation = cv2.INTER_CUBIC)
		img_resized = (img_resized[np.newaxis,:]/255.).astype('float32')
		
		prediction =  model.predict(img_resized)
		#print(f'Elapsed time for prediction: {timer() - start} s')
		return [prediction[0,0] , prediction[0,1]]

	except Exception as e:
		print('Mask detection: exception ! ')
		return [-1, -1]