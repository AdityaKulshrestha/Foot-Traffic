import cv2
import pickle
import socket
import struct

camera = False

if camera:
    vid = cv2.VideoCapture(0)
else:
    vid = cv2.VideoCapture('test_videos/test.mp4')
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# SERVER IP ADDRESS
host_ip = '127.0.0.1'  # Here enter the ip address according to your server
port = 12345

client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")

while client_socket and vid.isOpened():
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

