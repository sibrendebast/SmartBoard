#!/usr/bin/env python

import socket, threading
import math
import numpy as np
import time
import turtle
from graphics import *

global angles
global width
global height
global coordinate

width = 395*2;
height = 280*2;
win = GraphWin('Location',width,height)


angles = [float('nan'),float('nan'),float('nan'),float('nan')] #theta, phi, alpha, beta
coordinate = (float('nan',float('nan'))


def update_coor(angles,width,height,win):
    
    coordinate = Point(0,0)
    old_coor = Point(0,0)

    
    while True:
        nb_eq = 0
        A = np.zeros(shape=(4,2))
        y = np.zeros(shape=(4,1))
        if not math.isnan(angles[0]):
            B = np.array([[1, -math.tan(angles[0])]])
            z = np.array([[0]])
            A[0] = B
            y[0] = z
            nb_eq += 1
        if not math.isnan(angles[1]):
            B = np.array([[math.tan(angles[1]), 1]])
            z = np.array([[math.tan(angles[1])*width]])
            A[1] = B
            y[1] = z
            nb_eq += 1
        if not math.isnan(angles[2]):
            B = np.array([[math.tan(angles[2]), 1]])
            z = np.array([[height]])
            A[2] = B
            y[2] = z
            nb_eq += 1
        if not math.isnan(angles[3]):
            B = np.array([[-1, -math.tan(angles[3])]])
            z = np.array([[math.tan(angles[3])*height-width]])
            A[3] = B
            y[3] = z
            nb_eq += 1

        C = np.linalg.lstsq(A,y)
        
        coordinate = Point(int(C[0][0][0]),int(C[0][1][0]))
        line = Line(old_coor,coordinate)
        line.draw(win)
        #print coordinate            
    
        time.sleep(0.02)
        
        


                
            
class ServerThread(threading.Thread):
    

    def __init__(self,ip,port,angles):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket



    def run(self):    
        
        try:
            #clientsock.send('\nWelcome to the server\n')
            data = clientsock.recv(2048)
            #print data
            if ip == '10.0.7.119':
                #print ip, data,'\n'
                angles[1] = float(data)/180*math.pi
            elif ip == '10.0.7.198':
                #print ip, data,'\n'
                angles[0] = (float(data))/180*math.pi
            elif ip == '10.0.7.197':
                #print ip, data,'\n'
                angles[2] = (float(data))/180*math.pi
            
        except:
            pass
        



host = "10.0.7.119"
port = 12345

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

tcpsock.bind((host,port))

threads = []
run_event = threading.Event()
run_event.set()

t = threading.Thread(target=update_coor, args=(angles,width,height,win,))
threads.append(t)
t.start()




while True:
    tcpsock.listen(4)
    #print "\nListening for incoming connections..."
    (clientsock, (ip, port)) = tcpsock.accept()
    newthread = ServerThread(ip, port, angles,)
    newthread.start()
    threads.append(newthread)
    #print angles

    

for t in threads:
    t.join()


