import threading
import socket

class ClientThread(threading.Thread):
    

    def __init__(self,ip,port):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket



    def run(self):
        while True:
            try:
                clientsock.send('\nWelcome to the server\n')
                data = clientsock.recv(2048)
                #print data
                if ip == '169.254.155.157':
                    print ip, data
                    #angles[1] = float(data)/180*math.pi
                elif ip == '169.254.70.47':
                    print ip, data
                    #angles[0] = (float(data))/180*math.pi
                
            except:
                break
        



host = ""
port = 12345

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

tcpsock.bind((host,port))
threads = []



while True:
    try: 
        tcpsock.listen(4)
        #print "\nListening for incoming connections..."
        (clientsock, (ip, port)) = tcpsock.accept()
        newthread = ClientThread(ip, port)
        newthread.start()
        threads.append(newthread)
        #print angles
    except:
        for t in threads:
            t.join()
        break

for t in threads:
    t.join()
