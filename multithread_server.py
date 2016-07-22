#!/usr/bin/env python

import socket, threading
import math
import numpy as np
import time
import turtle



global angles
global width
global coor
global height

width = 395
height = 280
angles = [float('nan'),float('nan'),float('nan'),float('nan')] #theta, phi, alpha, beta
coor = (float('nan'),float('nan'))



class TurtleThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        wn = turtle.Screen()          # Creates a playground for turtles
        squirtle = turtle.Turtle()    # Create a turtle, assign to alex
        screen = squirtle.getscreen()
        screen.bgcolor('#000000')
        screen.setworldcoordinates(0,0,width,height)
        squirtle.speed(10)
        squirtle.pensize(4)
        squirtle.pencolor('#00FF00')

        while True:
            try:
                squirtle.goto(50,50)
                #squirtle.goto(coor[0],height-coor[1])
            except:
                pass
            time.sleep(0.03)
        
                
            
class ServerThread(threading.Thread):
    
    def __init__(self,ip,port):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket

    def run(self):
        try:
            data = clientsock.recv(2048)
            #print data
            if ip == '10.0.7.119':
                #print ip, data
                angles[1] = float(data)/180*math.pi
            elif ip == '10.0.7.198':
                #print ip, data
                angles[0] = (float(data))/180*math.pi
            elif ip == '10.0.7.197':
                #print ip, data
                angles[2] = (float(data))/180*math.pi
        except:
            pass
            
        
class CoordinateThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        print coor[0]
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
            if nb_eq < 2:
                coor = (float('nan'),float('nan'))
            else:
                C = np.linalg.lstsq(A,y)
                coor = (C[0][0][0],C[0][1][0])
            #print coordinate
            time.sleep(0.05)


host = "10.0.7.119"
port = 12345

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

tcpsock.bind((host,port))

threads = []

t = CoordinateThread()
threads.append(t)
t.start()

p = TurtleThread()
threads.append(p)
p.start()

while True:
    #print coor
    tcpsock.listen(1)
    (clientsock, (ip, port)) = tcpsock.accept()
    newthread = ServerThread(ip, port,)
    newthread.start()
    threads.append(newthread)

    

for t in threads:
    t.join()
