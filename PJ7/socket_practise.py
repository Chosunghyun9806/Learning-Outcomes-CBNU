import socket
import sys

""" FOR UDP """

# UDP_IP = "127.0.0.1"
# UDP_PORT = 5100
#
# udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
# udp_sock.bind((UDP_IP, UDP_PORT))
#
# while True:
#     data, addr = udp_sock.recvfrom(1024)
#     print(f"Received Data : {data.decode()}")
#     udp_sock.sendto(data, addr)
#     if data.decode() == 'x':
#         break
#
# udp_sock.close()
# sys.exit()

"""" FOR TCP """

TCP_IP = "127.0.0.1"
TCP_PORT = 5500

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.bind((TCP_IP, TCP_PORT))
tcp_socket.listen(100)

conn, addr = tcp_socket.accept()

print("Connect")

while True:
    data = conn.recv(1024)
    msg = data.decode()
    print(f"Received Data : {msg}")
    conn.sendall(msg.encode())

    if msg == 'x':
        tcp_socket.close()
        break

conn.close()
sys.exit()