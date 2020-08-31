#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import numpy as np
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import time
from absl import app, logging
import tensorflow as tf
from itertools import combinations
import math
# In[2]:


#Name of the class which we need to detect
classes = ["Person"]
output_path = './detections/'
# Initialize the parameters
confThreshold = 0.4  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image
count=0
# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Determine the output layer
layersNames = net.getLayerNames()
layers = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# In[3]:

def is_close(p1, p2):
    """
    #================================================================
    # 1. Purpose : Calculate Euclidean Distance between two points
    #================================================================    
    :param:
    p1, p2 = two points for calculating Euclidean Distance

    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    #=================================================================#
    return dst 


def convertBack(x, y, w, h): 
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
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
def socialDistance(img,classIds,confidences,boxes,confThreshold,nmsThreshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    centroid_dict = dict()
    objectId=0 
    for i in indices:
        i=i[0]
        box=boxes[i]
        x,y,w,h=box[0]+(box[2]/2),box[1]+(box[3]/2),box[2],box[3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox   
        # Append center point of bbox for persons detected.
        centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
        objectId += 1 #Increment the index for each detection           

    #=================================================================
    # 3.2 Purpose : Determine which person bbox are close to each other
    #=================================================================            	
    red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
    red_line_list = []
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Check the difference between centroid x: 0, y :1
        distance = is_close(dx, dy) 			# Calculates the Euclidean distance
        if distance < 75.0:						# Set our social distance threshold - If they meet this condition then..
            if id1 not in red_zone_list:
                red_zone_list.append(id1)       #  Add Id to a list
                red_line_list.append(p1[0:2])   #  Add points to the list
            if id2 not in red_zone_list:
                red_zone_list.append(id2)		# Same for the second id 
                red_line_list.append(p2[0:2])
    
    for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
        if idx in red_zone_list:   # if id is in red zone list
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Create Red bounding boxes  #starting point, ending point size of 2
        else:
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes
	#=================================================================#
	#=================================================================
	# 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
	#=================================================================        
    text = "People at Risk: %s" % str(len(red_zone_list)) 			# Count People at Risk
    location = (10,25)												# Set the location of the displayed text
    cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text
    for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
        start_point = red_line_list[check] 
        end_point = red_line_list[check+1]
        check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
        check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
        if (check_line_x < 75) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
            cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. 
    #=================================================================#
    return img,int(len(red_zone_list)) 



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


# In[19]:


# cap=cv2.VideoCapture("./crowdimage.jpg")
# print(cap.isOpened())
# vid_writer = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
#                             (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# print("hi")
# i=0
# while cap.isOpened():
#     _,frame=cap.read()
#     print(_)
#     if(_==False):
#         break
    
#     #Image PreProcessing
#     if(i%2==0):
#         blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    
#         count=0
#         net.setInput(blob)
#         outputs = net.forward(layers)
#         classIds,confidences,boxes=find_box_dimension(frame,outputs)
#         nonMaxSup(classIds,confidences,boxes,confThreshold,nmsThreshold)
#         cv2.putText(frame,  
#                     str(count),  
#                     (5,20),  
#                     cv2.FONT_HERSHEY_SIMPLEX, 1,  
#                     (0, 255, 255),  
#                     2,  
#                     cv2.LINE_4) 
#     #i+=1
#     #i=i%2
#     #cv2.putText(frame, str(count), (10,int(2*(frame.shape[1])/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
#     vid_writer.write(frame.astype(np.uint8))


# In[ ]:
