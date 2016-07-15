import socket

TCP_IP = ''
TCP_PORT = 12345
BUFFER_SIZE = 20

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
##s.setsockopt(socket.SOL_SOCKET, 25, b"eth0")
s.bind((TCP_IP,TCP_PORT))
s.listen(1)

conn,addr = s.accept()
print 'connection address:',addr
while 1:
    data = conn.recv(BUFFER_SIZE)
    if not data: break
    print 'received data:', data
    conn.send(data) #echo
conn.close()
