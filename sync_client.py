import socket
import threading
import time

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
                            s.send(received.split()[1])
                s.close()
            except:
                print 'connection failed'
                pass
        

    def stop(self):
        self.stopped = True


client = ClientThread('10.0.7.119',12345)
client.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    client.stop()

