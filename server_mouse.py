import socket
import threading
import time
import numpy as np
import math
import turtle
from numpy import *
from numpy.linalg import inv, det
from numpy.random import randn
import Tkinter as tk
from pymouse import PyMouse



global connections
global angles
global coordinate
global width
global heigth
global refresh_interval

refresh_interval = 0.01


#         theta        phi          alpha        beta
angles = [float('nan'),float('nan'),float('nan'),float('nan')]
coordinate = (float('nan'),float('nan'))

screen_width = 1920
screen_height = 1080
width = screen_width#395*2
height = screen_height#280*2


# A list containing all the active server connections
connections = []

#####    Class to hangle all the networking trafic
class ServerThread(threading.Thread):

    # initialize the serverThread
    def __init__(self,ip,port,socket):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket
        if ip == '10.0.7.119':
            self.angle = 1
        elif ip == '10.0.7.198':
            self.angle = 0
        elif ip == '10.0.7.197':
            self.angle = 2
                

    # send a request for data and wait for answer
    def send_request(self,request_num):
        self.socket.send('request '+str(request_num))
        answer = self.socket.recv(32)
        angles[self.angle] = float(answer.split()[1])/180.0*math.pi
        #print angles[self.angle]

    # stop the thread and close the socket
    def stop(self):
        self.socket.close()

    # return the IP address
    def get_ip(self):
        return self.ip
        
#####    Class that handles all the synchronization
class SyncThread(threading.Thread):

    # initilaize syncThread
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = True
        self.request_number = 0

    # ask open connections to send the angle with request_number
    def run(self):
        while self.running:
            #print self.request_number
            #loop over all open connections
            for con in connections:
                try:
                    #request data from all open connections
                    con.send_request(self.request_number)
                except:
                    print 'connection stopped',con.get_ip() 
                    con.stop()
                    connections.remove(con)
            #sleep for a bit (also wait for al the responses)
            time.sleep(refresh_interval)
            self.request_number += 1
            

    # set flag to stop the thread
    def stop(self):
        self.running = False

#####     Class that opens and closes TCP connections
class ConnectionHandler(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        host = "10.0.7.119"
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

##### Class that calculatues the coordinate of the marker based on the given angles
class CoordinateThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        
    def calc_coor(self):
        #initialize the needed variables
        nb_eq = 0
        A = np.zeros(shape=(4,2))
        y = np.zeros(shape=(4,1))
        #check if theta is set
        if not math.isnan(angles[0]):
            B = np.array([[1, -math.tan(angles[0])]])
            z = np.array([[0]])
            A[0] = B
            y[0] = z
            nb_eq += 1
        #check if phi is set
        if not math.isnan(angles[1]):
            B = np.array([[math.tan(angles[1]), 1]])
            z = np.array([[math.tan(angles[1])*width]])
            A[1] = B
            y[1] = z
            nb_eq += 1
        #check if alpha is set
        if not math.isnan(angles[2]):
            B = np.array([[math.tan(angles[2]), 1]])
            z = np.array([[height]])
            A[2] = B
            y[2] = z
            nb_eq += 1
        #check if beta is set
        if not math.isnan(angles[3]):
            B = np.array([[-1, -math.tan(angles[3])]])
            z = np.array([[math.tan(angles[3])*height-width]])
            A[3] = B
            y[3] = z
            nb_eq += 1
        # if we don't have enough equitations, we can't solve it
        if nb_eq < 3:
            #set coordinates to nan
            coordinate = (float('nan'),float('nan'))
        else:
            # using least squares to solve the system
            C = np.linalg.lstsq(A,y)
            coordinate = (C[0][0][0],C[0][1][0])
        return coordinate



def kf_predict(X, P, A, Q, B, U):
    X = dot(A,X) + dot(B,U)
    P = dot(A, dot(P, A.T)) + Q
    return (X,P)

def kf_update(X, P, Y, H, R):
    IM = dot(H,X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM,IS)
    return (X,P,K,IM,IS,LH)

def gauss_pdf(X,M,S):
    alpha = 0.5
    if M.shape[1]==1:
        DX = X - tile(M, X.shape[1])
        E = alpha * sum(DX.T * (dot(inv(S),DX)),axis = 0)
        E = E + 0.5 * M.shape[0] * log(2*pi) + (1-alpha) * log (det(S))
        P = exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1]) - M
        E = alpha * sum(DX.T * (dot(inv(S),DX)),axis = 0)
        E = E + 0.5 * M.shape[0] * log(2*pi) + (1-alpha) * log (det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = alpha * dot(DX.T , dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * log(2*pi) + (1-alpha) * log (det(S))
        P = exp(-E)

###############################################################################
##########                                                         ############
##########                            MAIN                         ############
##########                                                         ############
###############################################################################

### initiate connectionHandler
connHandler = ConnectionHandler()
connHandler.start()

### initiate syncThread
syncThread = SyncThread()
syncThread.start()

### initiate the calculation of the coordinates
coorCalculator = CoordinateThread()
coorCalculator.start()

### initialize the mouse controller
mouse = PyMouse()


######     initialize Kalman variables
dt = refresh_interval

# state vector
X = array([[0.0], [0.0], [0.0], [0.0]])
# covariance matrix
p = 1000
P = diag((p, p, p, p))
# state transition matrix
A = array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
# process noise matrix
Q = eye(X.shape[0])*0.5
B = eye(X.shape[0])
U = zeros((X.shape[0],1))

Y = array([[0],[0]])
H = array([[1,0,0,0],[0,1,0,0]])
R = eye(Y.shape[0])*7

# flag to keep the state of the mouse
pressed = False

while True:
    try:
        coordinate = coorCalculator.calc_coor()
        #print coordinate
        if not math.isnan(coordinate[0]):
            Y = array([[coordinate[0]],[coordinate[1]]])
            (X,P) = kf_predict(X,P,A,Q,B,U)
            (X,P,K,IM,IS,LH) = kf_update(X, P, Y, H, R)
            # move and press the mouse
            mouse.press((X[:2]).tolist()[0][0],(X[:2]).tolist()[1][0])
            pressed = True
        else:
            x = np.matrix('0. 0. 0. 0.').T 
            P = np.matrix(np.eye(4))*1000 # initial uncertainty
            # release the mouse
            if pressed:
                mouse.release((X[:2]).tolist()[0][0],(X[:2]).tolist()[1][0])
                pressed = False
            
        time.sleep(refresh_interval)
    # if we get a keyboard interrupt, kill all the running threads.
    except KeyboardInterrupt:
        connHandler.stop()
        syncThread.stop()
        for conn in connections:
            conn.stop()
        break
