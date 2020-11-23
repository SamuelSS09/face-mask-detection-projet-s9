from detect_mask import *


m = load_model('model.h5')



# for i in range(180,197):
# 	print(detect_mask(m,cv2.imread(f'temp_fig/{i:04d}.png')))


for i in range(0,20):
	print(detect_mask(m,cv2.imread(f'datasets/our_dataset/mask/{i:04d}.png')))