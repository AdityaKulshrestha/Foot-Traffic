import socket
import struct
import pickle
import cv2
import threading
from time import time
from main import track
import numpy as np
from utils import write_to_csv
from concurrent.futures import ThreadPoolExecutor


class VideoServer:
    def __init__(self, host, port, max_clients=5, timeout=30):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.timeout = timeout
        self.server_socket = None
        self.executor = ThreadPoolExecutor(max_workers=self.max_clients)
        self.terminate_flag = threading.Event()  # Event to signal termination

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_clients)
        print("Listening at", (self.host, self.port))

        while not self.terminate_flag.is_set():
            client_socket, client_addr = self.server_socket.accept()
            client_socket.settimeout(self.timeout)
            print(f"CLIENT {client_addr} CONNECTED!")
            self.executor.submit(self.handle_client, client_socket, client_addr)

    def handle_client(self, client_socket, client_addr):
        try:
            counter = 0
            new_counter = 0
            prev_trackids = []
            track_ids = []
            data = b""
            payload_size = struct.calcsize("Q")
            while not self.terminate_flag.is_set():
                loop_time = time()
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)
                    if not packet:
                        break
                    data += packet
                if not data:
                    break  # No data received, terminate the connection
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(4 * 1024)
                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = pickle.loads(frame_data)
                prev_trackids = track_ids.copy()
                img, counter, track_ids = track(frame, counter, track_ids)
                new_count = len(set(track_ids) - set(prev_trackids))
                print("New People Count - ", new_count)

                # print(f"{current_time} - New People count = {counter - prev_counter}")
                if new_count != 0:
                    write_to_csv(new_count)

                print("FPS =", 1 / (time() - loop_time))
                cv2.imshow("Process Video", img)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    self.terminate_flag.set()  # Set termination flag
                    cv2.destroyAllWindows()
                    client_socket.close()
                    print(f"CLIENT {client_addr} DISCONNECTED!")
                    break
        except socket.timeout:
            print(f"Connection timeout for client {client_addr}")
        except Exception as e:
            print(f"Error handling client {client_addr}: {e}")
        finally:
            client_socket.close()

    def stop(self):
        self.terminate_flag.set()  # Set termination flag
        self.server_socket.close()


def main():
    host_ip = '127.0.0.1'
    port = 12345
    server = VideoServer(host_ip, port)
    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
