# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:22:52 2020

@author: Sophie
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from detect_mask import *
from cv2 import VideoCapture
import time

detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt" , "res10_300x300_ssd_iter_140000.caffemodel")

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture("couvre-feu-a-paris-je-crois-que-je-vais-beaucoup-decoucher.mp3")

a = 0

model = load_model(weights_dir = 'weights/3classes_fineTunning_mobTrainable.h5',n_classes = 3,whole_model = True)

while True:
    a = a+1
    check, frame2 = video.read()
    #print(check)
    #print(frame2)

    base_img = frame2.copy()
    original_size = base_img.shape
    target_size = (300, 300)
    frame = cv2.resize(frame2, target_size)
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])  
    
    imageBlob = cv2.dnn.blobFromImage(image = frame)
    detector.setInput(imageBlob)
    faces = detector.forward()
    
    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(faces[0][0], columns = column_labels)
    
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence']>0.90]

    detections_df['left'] = (detections_df['left'] * 300).astype(int)
    detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
    detections_df['right'] = (detections_df['right'] * 300).astype(int)
    detections_df['top'] = (detections_df['top'] * 300).astype(int)
    
    for i, instance in detections_df.iterrows():
        confidence_score = round(100*instance["confidence"], 2)
        left = instance["left"]; right = instance["right"]
        bottom = instance["bottom"]; top = instance["top"]
        output = frame2[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x): int(right*aspect_ratio_x)]
        cv2.rectangle(frame2,(int(left*aspect_ratio_x),int(top*aspect_ratio_y)),(int(left*aspect_ratio_x)+int(right*aspect_ratio_x)-int(left*aspect_ratio_x),int(top*aspect_ratio_y)+int(bottom*aspect_ratio_y)-int(top*aspect_ratio_y)),(255,255,255),2)
        #print(output)
        if confidence_score>90:
            #A COMPLETER
            start = time.time()
            predictions = detect_mask_3_classes(model,output)
            print(f'Elapsed classification time {(time.time() - start):.8f} s')
            # predictions = detect_mask_3_classes_write(model,output,a)
            mask_case = np.argmax(predictions)
            if mask_case == 0:
                print(f'no mask, iteration {a}')
                upper_text = f'{100*predictions[0]:2.0f}% certainty of no mask presence'
                cv2.rectangle(frame2,(int(left*aspect_ratio_x),int(top*aspect_ratio_y)),(int(left*aspect_ratio_x)+int(right*aspect_ratio_x)-int(left*aspect_ratio_x),int(top*aspect_ratio_y)+int(bottom*aspect_ratio_y)-int(top*aspect_ratio_y)),(0,0,255),2)
                cv2.putText(frame2,upper_text,(int(left*aspect_ratio_x),int(top*aspect_ratio_y) - 10 ), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            elif mask_case ==  1:
                print(f'mask, iteration {a}')
                upper_text = f'{100*predictions[1]:2.0f}% certainty of mask presence'
                cv2.rectangle(frame2,(int(left*aspect_ratio_x),int(top*aspect_ratio_y)),(int(left*aspect_ratio_x)+int(right*aspect_ratio_x)-int(left*aspect_ratio_x),int(top*aspect_ratio_y)+int(bottom*aspect_ratio_y)-int(top*aspect_ratio_y)),(0,255,0),2)
                cv2.putText(frame2,upper_text,(int(left*aspect_ratio_x),int(top*aspect_ratio_y) -10 ), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            elif mask_case == 2:
                print(f'improper mask, iteration {a}')
                upper_text = f'{100*predictions[2]:2.0f}% certainty of improper mask presence'
                cv2.rectangle(frame2,(int(left*aspect_ratio_x),int(top*aspect_ratio_y)),(int(left*aspect_ratio_x)+int(right*aspect_ratio_x)-int(left*aspect_ratio_x),int(top*aspect_ratio_y)+int(bottom*aspect_ratio_y)-int(top*aspect_ratio_y)),(255,0,0),2)
                cv2.putText(frame2,upper_text,(int(left*aspect_ratio_x),int(top*aspect_ratio_y) - 10 ), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        roi_color = frame2[int(top*aspect_ratio_y):int(top*aspect_ratio_y)+int(bottom*aspect_ratio_y)-int(top*aspect_ratio_y), int(left*aspect_ratio_x):int(left*aspect_ratio_x)+int(left*aspect_ratio_x)-int(right*aspect_ratio_x)]         
    cv2.imshow("Capturing", frame2)
    

        
    #cv2.imshow("Capturing", output)
    

    #cv2.imshow("Capturing", output)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
print(a)
video.release()
cv2.destroyAllWindows()