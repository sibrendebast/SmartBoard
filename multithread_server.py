#!/usr/bin/env python

import socket, threading
import math
import numpy as np
import time
import turtle

global angles
global coordinate

angles = [float('nan'),float('nan'),float('nan'),float('nan')] #theta, phi, alpha, beta
coordinate = (float('nan'),float('nan'))

def update_coor(angles,coordinate):
    old_angles = [0,0,0,0]
    width = 100;
    height = 100;
    wn = turtle.Screen()      # Creates a playground for turtles
    squirtle = turtle.Turtle()    # Create a turtle, assign to alex
    squirtle.speed(10)
    squirtle.pensize(4)

    while 1:
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
        #print angles
        #print nb_eq
        C = np.linalg.lstsq(A,y)
        coordinate = (C[0][0],C[0][1])
        if nb_eq < 2:
            squirtle.penup()
        else:
            squirtle.goto((coordinate[0][0]-50)*5,(coordinate[1][0]-50)*5)
            squirtle.pendown()
        time.sleep(0.01)


                
            
class ClientThread(threading.Thread):
    

    def __init__(self,ip,port,angles):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket



    def run(self):    
        try:
            clientsock.send("\nWelcome to the server\n\n")
            data = clientsock.recv(2048)
            #print data
            if ip == '169.254.155.157':
                
                angles[0] = float(data)/180*math.pi
            elif ip == '169.254.70.47':
                angles[1] = float(data)/180*math.pi
        except:
            pass
        



host = ""
port = 12345

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

tcpsock.bind((host,port))
threads = []

t = threading.Thread(target=update_coor, args=(angles,coordinate,))
threads.append(t)
t.start()




while True:
    tcpsock.listen(4)
    #print "\nListening for incoming connections..."
    (clientsock, (ip, port)) = tcpsock.accept()
    newthread = ClientThread(ip, port, angles)
    newthread.start()
    threads.append(newthread)
    #print angles

for t in threads:
    t.join()


