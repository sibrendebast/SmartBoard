import socket
import threading
import time
import numpy as np
import math
from numpy import *
from numpy.linalg import inv, det
from numpy.random import randn
from pymouse import PyMouse
from scipy.optimize import fsolve
import pygame



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
width = screen_width
height = screen_height


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
            B = np.array([[1, -math.tan(angles[0]+delta_angles[0])]])
            z = np.array([[delta_y[0]*math.tan(angles[0]+delta_angles[0]) + delta_x[0]]])
            A[0] = B
            y[0] = z
            nb_eq += 1
        #check if phi is set
        if not math.isnan(angles[1]):
            B = np.array([[math.tan(angles[1]+delta_angles[1]), 1]])
            z = np.array([[math.tan(angles[1]+delta_angles[1])*(width+delta_x[1])-delta_y[1]]])
            A[1] = B
            y[1] = z
            nb_eq += 1
        #check if alpha is set
        if not math.isnan(angles[2]):
            B = np.array([[math.tan(angles[2]+delta_angles[2]), 1]])
            z = np.array([[height+delta_y[2]-delta_x[2]*math.tan(angles[2]+delta_angles[2])]])
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

#######          calibration
global delta_angles
global delta_x
global delta_y
    
delta_angles = [0,0,0,0]
delta_x = [0,0,0,0]
delta_y = [0,0,0,0]

delta_angles_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
delta_x_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
delta_y_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]

nb_vert_pnts = 1
nb_horz_pnts = 3 #HAS TO STAY 3!!!!

def calibrate():
    pygame.init()
    screen = pygame.display.set_mode((1920,1080),pygame.FULLSCREEN)
    width, height = screen.get_size()
    pygame.display.set_caption('Calibration')
    pygame.mouse.set_visible(False)

    green = (0,255,0)
    red = (255,0,0)
    white = (255,255,255)

    calibration_angles = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]

    
    for j in range(1,nb_vert_pnts+1):
        #print 'j', j
        for i in range(1,(nb_horz_pnts+1)):
            #print 'i', i
            screen.fill((0,0,0))
            pygame.draw.circle(screen, white, (i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)), 25)
            pygame.draw.lines(screen, red, False, [(i*width/(nb_horz_pnts+1)-30,j*height/(nb_vert_pnts+1)),(i*width/(nb_horz_pnts+1)+30,j*height/(nb_vert_pnts+1))], 3)
            pygame.draw.lines(screen, red, False, [(i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)-30),(i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)+30)], 3)
            pygame.display.update()
            # wait until there is a pen detected
            while str(angles[0]) == 'nan' or str(angles[1]) == 'nan' or str(angles[2]) == 'nan':
                try:
                    time.sleep(0.05)
                except:
                    pygame.quit()
                    raise

            #wait until we have enough data points
            
            data = [[],[],[],[]]
            while len(data[0]) < 40:
                try:
                    screen.fill((0,0,0))
                    pygame.draw.circle(screen, white, (i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)), int(25-0.5*len(data[0])))
                    pygame.draw.lines(screen, red, False, [(i*width/(nb_horz_pnts+1)-30,j*height/(nb_vert_pnts+1)),(i*width/(nb_horz_pnts+1)+30,j*height/(nb_vert_pnts+1))], 3)
                    pygame.draw.lines(screen, red, False, [(i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)-30),(i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)+30)], 3)
                    pygame.display.update()
                    if (not str(angles[0]) == 'nan' and not str(angles[1]) == 'nan' and not str(angles[2]) == 'nan'):
                        data[0].append(angles[0]) #theta
                        data[1].append(angles[1]) #phi
                        data[2].append(angles[2]) #alpha
                        data[3].append(angles[3]) #beta
                    time.sleep(0.05)
                except:
                    pygame.quit()
                    raise

            # wait until the pen is gone
            while not(str(angles[0]) == 'nan' or str(angles[1]) == 'nan' or str(angles[2]) == 'nan'):
                try:
                    screen.fill((0,0,0))
                    pygame.draw.lines(screen, green, False, [(i*width/(nb_horz_pnts+1)-30,j*height/(nb_vert_pnts+1)),(i*width/(nb_horz_pnts+1)+30,j*height/(nb_vert_pnts+1))], 3)
                    pygame.draw.lines(screen, green, False, [(i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)-30),(i*width/(nb_horz_pnts+1),j*height/(nb_vert_pnts+1)+30)], 3)
                    pygame.display.update()
                    time.sleep(0.05)
                except:
                    pygame.quit()
                    raise
            #print data
            calibration_angles[i-1] = [sum(data[0])/len(data[0]),sum(data[1])/len(data[1]),sum(data[2])/len(data[2]),sum(data[3])/len(data[3])]
            print calibration_angles[i-1]
                
        #print calibration_angles
        

        # theta
        x1, y1, theta1 =   width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[0][0]
        x2, y2, theta2 = 2*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[1][0]
        x3, y3, theta3 = 3*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[2][0]

        def equations_theta(p):
            theta,x,y = p
            return ( theta - math.atan2((x1 + x),(y1 + y)) + theta1, \
                      x     - (y2 + y)*math.tan(theta2 + theta) + x2,  \
                      y     - (x3 + x)/math.tan(theta3 + theta) + y3)

        delta_angles_list[j-1][0],delta_x_list[j-1][0],delta_y_list[j-1][0]=fsolve(equations_theta, (0,50,100))[:]
        print delta_angles_list[j-1][0],delta_x_list[j-1][0],delta_y_list[j-1][0]

        # phi
        x1, y1, theta1 =   width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[0][1]
        x2, y2, theta2 = 2*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[1][1]
        x3, y3, theta3 = 3*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[2][1]

        def equations_phi(p):
            dtheta,dx,dy = p
            return (  dtheta - math.atan2((y1 + dy),(width + dx - x1)) + theta1, \
                      dx     - (y2 + dy)/math.tan(theta2 + dtheta) - x2 + width,  \
                      dy     - (width - x3 + dx)*math.tan(theta3 + dtheta) + y3)
         
        delta_angles_list[j-1][1],delta_x_list[j-1][1],delta_y_list[j-1][1]=fsolve(equations_phi, (0,50,100))[:]
        print delta_angles_list[j-1][1],delta_x_list[j-1][1],delta_y_list[j-1][1]

        # alpha
        x1, y1, theta1 =   width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[0][2]
        x2, y2, theta2 = 2*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[1][2]
        x3, y3, theta3 = 3*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[2][2]

        def equations_alpha(p):
            dtheta,dx,dy = p
            return ( dtheta - math.atan2((height+ dy - y1),(x1 + dx)) + theta1, \
                      dx     - (height - y2 + dy)/math.tan(theta2 + dtheta) + x2,  \
                      dy     - (x3 + dx)*math.tan(theta3 + dtheta) + y3)

        delta_angles_list[j-1][2],delta_x_list[j-1][2],delta_y_list[j-1][2]=fsolve(equations_alpha, (0,50,100))[:]
        print delta_angles_list[j-1][2],delta_x_list[j-1][2],delta_y_list[j-1][2]

        # beta
        x1, y1, theta1 =   width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[0][3]
        x2, y2, theta2 = 2*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[1][3]
        x3, y3, theta3 = 3*width/(nb_horz_pnts+1), j*height/(nb_vert_pnts+1), calibration_angles[2][3]

        def equations_beta(p):
            theta,x,y = p
            return ( theta - math.atan2((x1 + x),(y1 + y)) + theta1, \
                      x     - (y2 + y)*math.tan(theta2 + theta) + x2,  \
                      y     - (x3 + x)/math.tan(theta3 + theta) + y3)

        delta_angles_list[j-1][3],delta_x_list[j-1][3],delta_y_list[j-1][3]=fsolve(equations_beta, (0,0,0))[:]
        print delta_angles_list[j-1][3],delta_x_list[j-1][3],delta_y_list[j-1][3]

    for i in range(0,4):
        som = 0
        for j in range(0,nb_vert_pnts):
            som += delta_angles_list[j][i]
        delta_angles[i] = som/(j+1)

    for i in range(0,4):
        som = 0
        for j in range(0,nb_vert_pnts):
            som += delta_x_list[j][i]
        delta_x[i] = som/(j+1)

    for i in range(0,4):
        som = 0
        for j in range(0,nb_vert_pnts):
            som += delta_y_list[j][i]
        delta_y[i] = som/(j+1)
    
    pygame.quit()

    
    
    print delta_angles
    print delta_x
    print delta_y

    

##### Kalman Filter implementation #####
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

### Wait for the data flow to start
time.sleep(1)

### Calibrate the system
calibrate()

##delta_angles = [-0.01489653883916651, 0.12564259946754294, 0.0718917376054, 0.0]
##delta_x = [45.313111630000272, 35.352680874347136, 35.1306102681, 0.0]
##delta_y =[85.720078716349605, 80.88786124506264, 105.569918538, 0.0]

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
        #print angles[0]
        if not math.isnan(coordinate[0]):
            Y = array([[coordinate[0]],[coordinate[1]]])
            (X,P) = kf_predict(X,P,A,Q,B,U)
            (X,P,K,IM,IS,LH) = kf_update(X, P, Y, H, R)
            # move and press the mouse
            #mouse.move((X[:2]).tolist()[0][0],(X[:2]).tolist()[1][0])
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
