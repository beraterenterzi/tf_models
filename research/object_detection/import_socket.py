import socket


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('169.254.156.17', 6060))
s.listen(5)

while True:
    clientSocket, address = s.accept()
    print("Connection adress:", address)
    clientSocket.send(bytes("haluk", "utf-8"))
    clientSocket.close()