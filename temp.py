from detect_mask import *
from PIL import Image
import cv2

model = load_model(weights_dir = 'weights/MobileNetV2_n_dataset.h5')

img = np.array( Image.open('datasets/n_dataset/no_mask/0000.png') )
img_resized = cv2.resize(img, (128, 128) ,interpolation = cv2.INTER_CUBIC)
img_resized = (img[:]/255.).astype('float32')
model.predict(img)