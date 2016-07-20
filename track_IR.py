#####           Import necessary packages      #####
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import threading
import numpy as np
import socket
from threading import Thread

                  
#####                  CONSTANTS                #####
FOV = 62
TCP_IP = '10.0.7.119'
TCP_PORT = 12345
BUFFER_SIZE = 1024

WIDTH = 640
HEIGHT = 480
SIDEWIDTH = 100

global message
message = 'nan'

####### initialize the camera and grab a reference to the ram camera capture
##camera = PiCamera()
##camera.resolution = (WIDTH,HEIGHT)
###camera.iso = 50
##camera.framerate = 32
##rawCapture = PiRGBArray(camera, size=(WIDTH,HEIGHT))
##stream = camera.capture_continuous(rawCapture, format="bgr",use_video_port=True)
##
#######        Allow the camera to warmup          #####
##time.sleep(0.1)

#####   class to put video recording on seperate thread   #####
class PiVideoStream:
    def __init__(self, resolution=(WIDTH, HEIGHT), framerate=32):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return
            
    def read(self):
	# return the frame most recently read
	return self.frame
 
    def stop(self):
	# indicate that the thread should be stopped
	self.stopped = True

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
        #print 'begin sending'
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.send(message)
        #print message
        s.close()
    except:
        #print 'sending failed'
        pass

##### Start client thread to send data to server #####
##threads = []
##client = ClientThread(TCP_IP,TCP_PORT,message)
##client.start()
##threads.append(client)


#####      Capture frames from the camera        #####
vs = PiVideoStream().start()
time.sleep(0.5)

while True:
    image = vs.read()
    image = image[250:265, SIDEWIDTH:WIDTH-SIDEWIDTH,0]
    
    
    if image.max() > 200:
        x=np.unravel_index(image.argmax(),image.shape)[1]
        theta = 148*(float(x)+SIDEWIDTH)/WIDTH-26.245
        #print theta
    else:
        theta = float('nan')

    
##    ret,thresh = cv2.threshold(image,170,255,0)
##    im, contours,hierarchy = cv2.findContours(thresh, 1, 2)
##    try:
##        M = cv2.moments(contours[0])
##        x = (M['m10']/M['m00'])
##        #print x, np.unravel_index(image.argmax(),image.shape)[1]
##        theta = 148*(x+SIDEWIDTH)/WIDTH-26.245
##    except:
##        theta = float('nan')
##        pass

    send_data(str(theta))
    #print  theta
    #cv2.imshow('dokljshfj',image)
    time.sleep(0.02)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()






    

