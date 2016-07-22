import socket
import threading
import time

global connections

connections = []

class ServerThread(threading.Thread):
    
    def __init__(self,ip,port,socket):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket

    def send_request(self,request_num):
        self.socket.send('request '+str(request_num))
        answer = self.socket.recv(32)
        print answer.split()[1]

    def stop(self):
        self.socket.close()

    def get_ip(self):
        return self.ip
        

class SyncThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.running = True
        self.request_number = 0

    def run(self):
        while self.running:
            print self.request_number
            for con in connections:
                try:
                    con.send_request(self.request_number)
                except:
                    print 'connection stopped',con.get_ip() 
                    con.stop()
                    connections.remove(con)                    
            self.request_number += 1
            time.sleep(0.033)

    def stop(self):
        self.running = False


class ConnectionHandler(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        host = "10.0.7.119"
        port = 12345
        self.running = True

        self.tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.tcpsock.bind((host,port))

    def run(self):
        while self.running:
            print 'listening for connections'
            self.tcpsock.listen(3)
            (sock, (ip, port)) = self.tcpsock.accept()
            print 'connection from '+str(ip)
            conn = ServerThread(ip, port, sock,)
            conn.start()
            connections.append(conn)

    def stop(self):
        self.running = False


connHandler = ConnectionHandler()
connHandler.start()

syncThread = SyncThread()
syncThread.start()



while True:
    try:
        time.sleep(0.01)
    except KeyboardInterrupt:
        connHandler.stop()
        syncThread.stop()
        for conn in connections:
            conn.stop()
        raise
