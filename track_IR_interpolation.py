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
SIDEWIDTH = 50

global message
global theta
global old_theta
global delta_time


message = 'nan'
theta = float('nan')
old_theta = float('nan')
timestamp = time.time()
delta_time = 0


##### initialize the camera and grab a reference to the ram camera capture
camera = PiCamera()
camera.resolution = (WIDTH,HEIGHT)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(WIDTH,HEIGHT))


#####        Allow the camera to warmup          #####
time.sleep(0.1)

#####     Client Class to send out the data      #####
class ClientThread(threading.Thread):
    

    def __init__(self,ip,port,message):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket

    def run(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((TCP_IP, TCP_PORT))
                while True:
                    #print s.recv(1024)
                    #print message
                    s.send(message)
                    time.sleep(0.033)
                s.close()
            except KeyboardInterrupt:
                break
            except:
                pass
            
        

##def send_data(message):
##    try:
##        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##        s.connect((TCP_IP, TCP_PORT))
##        s.send(message)
##        s.close()
##    except:
##        pass

##### Start client thread to send data to server #####
threads = []
client = ClientThread(TCP_IP,TCP_PORT,message)
client.start()
threads.append(client)

##### A class to interpolate the angles in a different thread #####
class InterpolationThread(threading.Thread):
    
    def __init__(self,message,theta,old_theta,delta_time):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            delta_theta = theta - old_theta
            print delta_theta
            time.sleep(0.05)

interpolator = InterpolationThread(message,theta,old_theta,delta_time)
interpolator.start()
threads.append(interpolator)

#####      Capture frames from the camera        #####
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array
    delta_time = time.time()-timestamp
    timestamp = time.time()
    #print deltatime
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = image[250:265, SIDEWIDTH:WIDTH-SIDEWIDTH]

    ret,thresh = cv2.threshold(image,170,255,0)
    im, contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    try:
        M = cv2.moments(contours[0])
        x = (M['m10']/M['m00'])
        old_theta = theta
        theta = 148*(x+SIDEWIDTH)/WIDTH-26.245
        
    except:
        old_theta = theta
        theta = float('nan')

    #print old_theta, theta  
    
    #send_data(str(theta))
    #print  theta
    #cv2.imshow('dokljshfj',image)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break


for t in threads:
        t.join()





    

