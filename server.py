import socket
import struct
import pickle
from ultralytics import YOLO
import cv2
import threading
from sort import *
import numpy as np
from time import time
from main import count_people

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Setting up IP Address and port name
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
port = 12345
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at", socket_address)

# Setting up the detecting configs
model = YOLO("MyProject/model/yolov8s.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [0, 300, 3000, 800]
limitsDown = [0, 300, 3000, 800]

totalCountUp = []
totalCountDown = []


def show_client(addr, client_socket):
    # try:
    print('CLIENT {} CONNECTED!'.format(addr))
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

            img = count_people(frame)
            print("FPS = ",1/ (time() - loop_time))
            loop_time = time()

            cv2.imshow("Process Video", img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


while True:
    client_socket, addr = server_socket.accept()
    thread = threading.Thread(target=show_client, args=(addr, client_socket))
    thread.start()
    print("TOTAL CLIENTS ", threading.activeCount() - 1)
