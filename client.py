import cv2
import pickle
import socket
import struct

import imutils

camera = False

if camera:
    vid = cv2.VideoCapture(0)
else:
    vid = cv2.VideoCapture('MyProject/test.mp4')
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# SERVER IP ADDRESS
host_ip = '192.168.0.167'  # Here enter the ip address according to your server
port = 12345

client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")

if client_socket:
    while vid.isOpened():
        try:
            img, frame = vid.read()
            # Convert frames in to bytes using pickle
            a = pickle.dumps(frame)
            # Bytes packed with Q format
            message = struct.pack("Q", len(a)) + a
            # Send this message
            client_socket.sendall(message)
        except:
            print("VIDEO FINISHED!")
            break

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
