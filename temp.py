from detect_mask import *
from PIL import Image
import cv2
from detect_mask import *


a = 0

model = load_model(weights_dir = 'weights/3classes_simulatedReduced_MobTrainable.h5',n_classes = 3,whole_model = True)
img = np.array( Image.open('datasets/3classes_simulated_dataset/wrong_ported_mask/0010.png') )
img_resized = cv2.resize(img, (128, 128) ,interpolation = cv2.INTER_CUBIC)
img_resized = (img_resized[np.newaxis,:]/255.).astype('float32')
model.predict(img_resized)
