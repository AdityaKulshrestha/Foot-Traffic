import socket
import struct
import pickle
import threading
import queue
from time import time
import cv2
from main import count_people


class FrameProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue()
        self.running = True

    def process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                img = count_people(frame)
                cv2.imshow("Processed Video", img)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    self.running = False
                    cv2.destroyAllWindows()
                    break
                self.frame_queue.task_done()

    def add_frame(self, frame):
        self.frame_queue.put(frame)


def show_client(addr, client_socket, frame_processor):
    print('CLIENT {} CONNECTED!'.format(addr))
    if client_socket:
        data = b""
        payload_size = struct.calcsize("Q")
        while True:
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
            frame_processor.add_frame(frame)
            print("FPS = ", 1 / (time() - loop_time))
            loop_time = time()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    port = 12345
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)
    server_socket.listen()
    print("Listening at", socket_address)

    frame_processor = FrameProcessor()
    process_thread = threading.Thread(target=frame_processor.process_frames)
    process_thread.start()

    while True:
        client_socket, addr = server_socket.accept()
        thread = threading.Thread(target=show_client, args=(addr, client_socket, frame_processor))
        thread.start()
        print("TOTAL CLIENTS ", threading.activeCount() - 1)

    frame_processor.running = False
    frame_processor.frame_queue.join()
    process_thread.join()
