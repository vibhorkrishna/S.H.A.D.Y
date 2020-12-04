#================================================================ 
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations
import pafy
import youtube_dl
#import image_email_car   # Uncomment for alert to email

def is_close(p1, p2):
    """
    #================================================================
    # Purpose : Calculate Euclidean Distance between two points
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
    # Purpose : Converts center coordinates to rectangle coordinates
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

#alert_var = 0      # Uncomment for alert to email       # makes sure that alert (Sending an E-mail) is generated only once

def cvDrawBoxes(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet

    :return:
    img with bbox
    """
    #global alert_var
    
    #================================================================
    # Purpose : Filter out Cars class from detections and get 
    #           bounding box centroid for each car detection.
    #================================================================
    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0
        for detection in detections:				# In this if statement, we filter all the detections for cars only
            # Check for the only car name tag 
            name_tag = str(detection[0].decode())   # Coco file has string of all the names
            if name_tag == 'car':                
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	# Store the center points of the detections
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
                # Append center point of bbox for cars detected.
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
                objectId += 1 #Increment the index for each detection      
    #=================================================================#
    
    #=================================================================
    # Purpose : Determine which car bbox are close to each other
    #=================================================================            	
        vehicle_red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
        vehicle_red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
            #dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Check the difference between centroid x: 0, y :1
            #distance = is_close(dx, dy) 			# Calculates the Euclidean distance
            
            #if distance < 50.0:						# Set our distance threshold - If they meet this condition then..
            
            if not ((p1[2]>=p2[4]) or (p1[4]<=p2[2]) or (p1[5]<=p2[3]) or (p1[3]>=p2[5])):
                if id1 not in vehicle_red_zone_list:
                    vehicle_red_zone_list.append(id1)       #  Add Id to a list
                    vehicle_red_line_list.append(p1[0:2])   #  Add points to the list
                if id2 not in vehicle_red_zone_list:
                    vehicle_red_zone_list.append(id2)		# Same for the second id 
                    vehicle_red_line_list.append(p2[0:2])
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in vehicle_red_zone_list:   # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Create Red bounding boxes  #starting point, ending point size of 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes
		#=================================================================#

		#=================================================================
    	# Purpose : Displaying the results and sending an alert message
    	#=================================================================        
        if len(vehicle_red_zone_list)!=0:
            text = "Crash Detected"
            
            #Uncomment the below lines for alert to email
            #if alert_var == 8:         # makes sure that alert is generated when there are atleast 3 frames which shows that a fall has been detected
            #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #    cv2.imwrite('crash_alert.jpg',img)
            #    image_email_car.SendMail('crash_alert.jpg')
            #alert_var += 1;
        
        else:
            text = "Crash Not Detected"
            #alert_var = 0     #Uncomment for alert to email           # makes sure that alert is generated when there are 3 simultaeous frames of fall detection
            
        location = (10,25)												# Set the location of the displayed text
        if len(vehicle_red_zone_list)!=0:
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)  # Display Text
        else:
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # Display Text
            
    return img


netMain = None
metaMain = None
altNames = None

def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4-tiny.cfg"                                 # Path to cfg
    weightPath = "./yolov4-tiny.weights"                                 # Path to weights
    metaPath = "./cfg/coco.data"                                         # Path to meta data
    if not os.path.exists(configPath):                                   # Checks whether file exists otherwise return ValueError  
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                                  # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)                  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
            
    #cap = cv2.VideoCapture(0)                                           # Uncomment to use Webcam
    
    #cap = cv2.VideoCapture("crash4.mp4")                                # Uncomment for Local Stored video detection - Set input video
    
    #url = "https://www.youtube.com/watch?v=isveXCH4NcM"                 # Uncomment these lines for video from youtube
    #video = pafy.new(url)
    #best = video.getbest(preftype="mp4")
    #cap = cv2.VideoCapture()
    #cap.open(best.url)    
    
    #cap = cv2.VideoCapture('http://192.168.0.106:4747/mjpegfeed')       # Uncomment for Video from Mobile Camera (DroidCam Hosted Camera)
    
    frame_width = int(cap.get(3))                                        # Returns the width and height of capture video   
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    #print("Video Reolution: ",(width, height))

    #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,  # Uncomment to save the output video    # Set the Output path for video writer
            #(new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)         # Create image according darknet for compatibility of network
    
    while True:                                                          # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                   
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)          # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())    # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change
        image = cvDrawBoxes(detections, frame_resized)                   # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))                             # Prints frames per second
        cv2.imshow('Demo', image)                                    # Display Image window
        cv2.waitKey(3)
        #out.write(image)                                            # Write that frame into output video
        
    cap.release()                                                    # For releasing cap and out. 
    #out.release()                                                   # Uncomment to save the output video 
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()                                                           # Calls the main function YOLO()
