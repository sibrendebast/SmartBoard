## Script to read out the camera of a raspberry pi and and calculate the angle
## at which an IR pen is located in relation to the raspberry pi. The raspberry
## pi is fitted with an IR camera (version 2). The program listens to incoming
## connections to send out the measured angle.
##
## Author: Sibren De Bast
## in assignment of Easics,
## July 2016


###############################################################################
##########                                                         ############
##########                      DEPENDENCIES                       ############
##########                                                         ############
###############################################################################

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import threading
import numpy as np
import socket
from threading import Thread
                  
###############################################################################
##########                                                         ############
##########                 CONSTANTS AND VARIABLES                 ############
##########                                                         ############
###############################################################################

global connections
connections = []

## resolution of the camera
WIDTH = 1296
HEIGHT = 736
## How much of the sides we don't need
SIDEWIDTH = WIDTH/8

## variable to hold the angle at which the pen is located
global theta
theta = float('nan')

###############################################################################
##########                                                         ############
##########                     CLASS DEFINITIONS                   ############
##########                                                         ############
###############################################################################

#####   class to put video recording on seperate thread   #####
class PiVideoStream:
    def __init__(self, resolution=(WIDTH, HEIGHT), framerate=32):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                format="rgb", use_video_port=True)

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


#####     Class that opens and closes TCP connections
class ConnectionHandler(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        host = ""
        port = 12345
        self.running = True
        # initiate the TCP connection
        self.tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcpsock.bind((host,port))

    # thread keeps listening for incoming connections
    def run(self):
        while self.running:
            # listen for incoming connection
            print 'listening for connections'
            self.tcpsock.listen(3)
            (sock, (ip, port)) = self.tcpsock.accept()
            print 'connection from '+str(ip)
            # start serverthread to handle incoming connection
            conn = ServerThread(ip, port, sock,)
            conn.start()
            connections.append(conn)

    # set flag to stop the thread
    def stop(self):
        self.running = False

        
#####    Class to handle all the networking trafic
class ServerThread(threading.Thread):

    # initialize the serverThread
    def __init__(self,ip,port,socket):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket
        self.stopped = False

    def run(self):
        while not self.stopped:
            received = self.socket.recv(128)
            #print received
            ## when receiving the right signal, send back the angle
            if received.split()[0] == 'request':
                self.socket.send(received.split()[1]+' '+str(theta))
        
                        
    
    # stop the thread and close the socket
    def stop(self):
        self.stopped = True
        self.socket.close()

    # return the IP address
    def get_ip(self):
        return self.ip

###############################################################################
##########                                                         ############
##########                            MAIN                         ############
##########                                                         ############
###############################################################################
        
### initiate connectionHandler
connHandler = ConnectionHandler()
connHandler.start()

##      Capture frames from the camera     
vs = PiVideoStream().start()

## Give the camera some time to start
time.sleep(0.5)

##try:
while True:
    image = vs.read()
    ## crop the image to the needed size
    image = image[300:400, SIDEWIDTH:WIDTH-SIDEWIDTH,0]
    
    if image.max() > 230:
        x=np.unravel_index(image.argmax(),image.shape)[1]
        theta = 90-(131.1*(float(x)+SIDEWIDTH)/WIDTH-20.642)
    else:
        theta = float('nan')
    #print theta
    time.sleep(0.03)

       
        
##except KeyboardInterrupt:
##    vs.stop()
##    connHandler.stop()    
##    for conn in connections:
##        conn.stop()
