#####           Import necessary packages      #####
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import threading
import numpy as np
import socket

                  
#####                  CONSTANTS                #####
FOV = 62
TCP_IP = '169.254.155.157'
TCP_PORT = 12345
BUFFER_SIZE = 1024

WIDTH = 640
HEIGHT = 480

global message
message = 'nan'

##### initialize the camera and grab a reference to the ram camera capture
camera = PiCamera()
camera.resolution = (WIDTH,HEIGHT)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(WIDTH,HEIGHT))


#####        Allow the camera to warmup          #####
time.sleep(0.1)

#####     Client Class to send out the data      #####
##class ClientThread(threading.Thread):
##    
##
##    def __init__(self,ip,port,message):
##        threading.Thread.__init__(self)
##        self.ip = ip
##        self.port = port
##        self.socket = socket
##
##
##
##    def run(self):
##        try:
##            while True:
##                try:
##                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##                    s.connect((TCP_IP, TCP_PORT))
##                    while True:
##                        #print s.recv(1024)
##                        s.send(message)
##                        time.sleep(0.033)
##                    s.close()
##                except:
##                    pass

def send_data(message):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.send(message)
        s.close()
    except:
        pass

##### Start client thread to send data to server #####
##threads = []
##client = ClientThread(TCP_IP,TCP_PORT,message)
##client.start()
##threads.append(client)


#####      Capture frames from the camera        #####
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = image[260:265, 0:WIDTH]

    ret,thresh = cv2.threshold(image,170,255,0)
    im, contours,hierarchy = cv2.findContours(thresh, 1, 2)
       
    theta = float('nan')

    try:
        M = cv2.moments(contours[0])
        x = (M['m10']/M['m00'])
        theta = 148*x/WIDTH-26.245
        if theta < -10  or theta > 100:
            theta = float('nan')
    except:
        pass
    
    send_data(str(theta))
    #print  theta
    #cv2.imshow('dokljshfj',image)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break


for t in threads:
        t.join()





    

