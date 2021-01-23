# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:49:12 2021

@author: Sophie
"""

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import usefull_functions
import numpy as np


frozen_graph="C:/Users/Sophie/Downloads/frozen.pb"


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile(frozen_graph, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



cap = cv2.VideoCapture(0)
with detection_graph.as_default():
    with tf.compat.v1.Session() as sess:
        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
              tensor_name)
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = usefull_functions.run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            usefull_functions.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
               {1: {'id': 1, 'name': 'face'}},
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)
            cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        
    #cv2.imshow("Capturing", output)
    

    #cv2.imshow("Capturing", output)
    

    
print(a)
cap.release()
cv2.destroyAllWindows()