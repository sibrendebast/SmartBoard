# import necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
##import client
import socket


##CONSTANTS
FOV = 62
TCP_IP = '169.254.155.157'
TCP_PORT = 12345
BUFFER_SIZE = 1024

WIDTH = 1920
HEIGHT = 208

def send_data(message):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.send(message)
    s.close()

#initialize the camera and grab a reference to the ram camera capture
camera = PiCamera()
camera.resolution = (WIDTH,HEIGHT)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(WIDTH,HEIGHT))

# allow the camera to warmup
time.sleep(0.1)

##low_line = 0
##high_line = 368

lowerIR = np.array([220,220,220])
upperIR = np.array([255,255,255])

#capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array
    #crop image
    image = image[150:200, 0:WIDTH]
    #flip the image
    #image = cv2.flip(image,0)

    

    mask = cv2.inRange(image, lowerIR, upperIR)
    
    im2, contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##    cv2.drawContours(image, contours, -1, (0,0,255), -1)

    theta = float('nan')
    
    try:
        cnt = contours[0]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        theta = 148*x/WIDTH-26.245
        #print x,theta
        radius = int(radius)
        #image = cv2.circle(image,center,radius,(0,255,0),2)
##        low_line =  int(y)-20
##        high_line = int(y)+20
    except:
        pass

    try:
        send_data(str(theta))
    except:
        pass
    print  theta
    cv2.imshow('Frame',image)
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord('q'):
        break





