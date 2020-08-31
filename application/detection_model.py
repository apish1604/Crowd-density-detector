#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import numpy as np
import os
import tensorflow as tf

# In[2]:


#Name of the class which we need to detect
classes = ["Person"]
#output_path = './detections/'
# Initialize the parameters
confThreshold = 0.4  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image
count=0
# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('/home/ksangwan/Documents/flask-application/final_minor_project/application/yolov3.cfg', '/home/ksangwan/Documents/flask-application/final_minor_project/application/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Determine the output layer
layersNames = net.getLayerNames()
layers = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# In[3]:

# Draw the predicted bounding box
def drawPred(img,classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


# In[4]:


# Remove the bounding boxes with low confidence using non-maxima suppression
def nonMaxSup(img,classIds,confidences,boxes,confThreshold,nmsThreshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    global count
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(img,classIds[i], confidences[i], left, top, left + width, top + height)
        #print(classIds[i])
        if classIds[i] in [0]:  
            count = count + 1
    return count

# In[5]:


# Scan through all the bounding boxes output from the network and keep only the
# ones with high confidence scores. Assign the box's class label as the class with the highest score.
def find_box_dimension(frame,outputs):
    height=frame.shape[0]
    width=frame.shape[1]
    classIds=[]
    confidences=[]
    boxes=[]
    
    for output in outputs:
        for detect in output:
            scores=detect[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence>confThreshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                left = int(center_x - w / 2)
                top = int(center_y - h / 2)
                if classId < 1:
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, w, h])
    return classIds,confidences,boxes



