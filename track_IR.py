#####           Import necessary packages      #####
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import threading
import numpy as np
import socket
from threading import Thread

                  
#####                  CONSTANTS                #####
TCP_IP = '10.0.7.119'
TCP_PORT = 12345
BUFFER_SIZE = 1024

WIDTH = 1296
HEIGHT = 736
SIDEWIDTH = WIDTH / 10

global theta
theta = float('nan')

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
class ClientThread(threading.Thread):
    

    def __init__(self,ip,port):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.stopped = False


    def run(self):
        while not self.stopped:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #print 'connecting....'
                s.connect((self.ip, self.port))
                while not self.stopped:
                        received = s.recv(128)
                        #print received
                        if received.split()[0] == 'request':
                            #print received.split()[1]
                            s.send(received.split()[1]+' '+str(theta))
                s.close()
            except:
                #print 'connection failed'
                time.sleep(0.1)
                pass
        

    def stop(self):
        self.stopped = True


client = ClientThread('10.0.7.119',12345)
client.start()

#####      Capture frames from the camera        #####
vs = PiVideoStream().start()
time.sleep(0.5)

try:
    while True:
        image = vs.read()
        image = image[300:400, SIDEWIDTH:WIDTH-SIDEWIDTH,0]
        
        
        if image.max() > 220:
            x=np.unravel_index(image.argmax(),image.shape)[1]
            #theta = 148*(float(x)+SIDEWIDTH)/WIDTH-26.245
            theta = 90-(131.1*(float(x)+SIDEWIDTH)/WIDTH-20.642)
            #print theta
        else:
            theta = float('nan')
        
        time.sleep(0.02)

       
        
except KeyboardInterrupt:
    vs.stop()
    client.stop()








    

