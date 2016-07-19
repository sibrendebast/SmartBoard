import socket

def send_data(message):
    TCP_IP = '169.254.155.157'
    TCP_PORT = 12345
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    while True:
        try:
            print s.recv(1024)
            s.send(message)
        except:
            break
    s.close()




send_data('80')
