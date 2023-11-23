import socket
import sys

""" FOR UDP """

# UDP_PORT = 5100
# ad = input("IP Addr : ")
#
# udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
# while True:
#     send_data = input("Message: ")
#     udp_socket.sendto(send_data.encode(), (ad, UDP_PORT))
#
#     data, addr = udp_socket.recvfrom(1024)
#     print(f"Received Data : {data.decode()}")
#
#     if data.decode() == 'x':
#         break
#
# udp_socket.close()
# sys.exit()

""" FOR TCP """
12
TCP_PORT = 5500
ad = input("IP Addr : ")

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

tcp_socket.connect((ad, TCP_PORT))

while True:
    send_data = input("Message: ")
    tcp_socket.send(send_data.encode())

    data = tcp_socket.recv(1024)
    print(f"Received Data : {data.decode()}")

    if data.decode() == 'x':
        break

tcp_socket.close()
sys.exit()

