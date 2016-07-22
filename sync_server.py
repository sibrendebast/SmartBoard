import socket
import threading
import time

global connections
global angles

# A list containing all the active server connections
connections = []
#         theta        phi          alpha        beta
angles = [float('nan'),float('nan'),float('nan'),float('nan')]


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
        angles[self.angle] = answer.split()[1]
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
            time.sleep(0.033)
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
        if nb_eq < 2:
            #set coordinates to nan
            coor = (float('nan'),float('nan'))
        else:
            # using least squares to solve the system
            C = np.linalg.lstsq(A,y)
            coor = (C[0][0][0],C[0][1][0])
        print coordinate

### initiate connectionHandler
connHandler = ConnectionHandler()
connHandler.start()

### initiate syncThread
syncThread = SyncThread()
syncThread.start()

### initiate the calculation of the coordinates
coorCalculator = CoordinateThread()
coorCalculator.start()

### keep the main thread running until we want to stop all threads
while True:
    try:
        time.sleep(0.01)
    # if we get a keyboard interrupt, kill all the running threads.
    except KeyboardInterrupt:
        connHandler.stop()
        syncThread.stop()
        for conn in connections:
            conn.stop()
        raise
