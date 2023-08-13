import socket
import struct
import pickle
from ultralytics import YOLO
import cv2
import threading
import numpy as np
from time import time
from main import count_people, track

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Setting up IP Address and port name. Automatically detects the current ip address.
# host_name = socket.gethostname()
# host_ip = socket.gethostbyname(host_name)

# Custom host address
host_ip = '127.0.0.1'

port = 12345
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at", socket_address)


def show_client(addr, client_socket):
    # try:
    print('CLIENT {} CONNECTED!'.format(addr))
    counter = 0
    global track_ids
    track_ids = []
    if client_socket:
        data = b""
        payload_size = struct.calcsize("Q")
        while True:
            loop_time = time()
            while len(data) < payload_size:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    break
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4 * 1024)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)

            # img = count_people(frame) #Working currently
            img, counter = track(frame, counter, track_ids)
            # img, counter = Tracker().detect_persons(frame)

            print("New People count = ", counter)
            print("FPS = ", 1 / (time() - loop_time))

            cv2.imshow("Process Video", img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                client_socket.close()  # Close the connection
                print('CLIENT {} DISCONNECTED!'.format(addr))
                break


while True:
    client_socket, addr = server_socket.accept()
    thread = threading.Thread(target=show_client, args=(addr, client_socket))
    thread.start()
    print("TOTAL CLIENTS ", threading.activeCount() - 1)
