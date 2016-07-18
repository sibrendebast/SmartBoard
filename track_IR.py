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

WIDTH = 640
HEIGHT = 480

def send_data(message):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.send(message)
    s.close()

#initialize the camera and grab a reference to the ram camera capture
camera = PiCamera()
camera.resolution = (WIDTH,HEIGHT)
camera.framerate = 32
#camera.color_effects = (128,128)
rawCapture = PiRGBArray(camera, size=(WIDTH,HEIGHT))

# allow the camera to warmup
time.sleep(0.1)


lowerIR = np.array([150,150,150])
upperIR = np.array([255,255,255])

#capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = image[260:265, 0:WIDTH]

    ret,thresh = cv2.threshold(image,127,255,0)
    im, contours,hierarchy = cv2.findContours(thresh, 1, 2)
       
    theta = float('nan')

    
    try:
        M = cv2.moments(contours[0])
        x = (M['m10']/M['m00'])
        theta = 148*x/WIDTH-26.245
    except:
        pass



    try:
        send_data(str(theta))
    except:
        pass

    
    print  theta

    
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord('q'):
        break





