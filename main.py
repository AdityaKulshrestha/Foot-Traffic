from ultralytics import YOLO
import cv2
import random
import math
import logging
import numpy as np
from tracker import Tracker

# Setting up  logging
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

model = YOLO("model/yolov8n.pt")

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


# Not in use currently
def count_people(img=None):
    results = model(img, stream=True)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100
            if currentClass == "person" and conf > 0.8:
                continue

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            detections.append([[x1, y1, w, h], conf, 0])

    tracks = tracker.update_tracks(detections, frame=img)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + 20, ymin), (0, 255, 0), -1)
        cv2.putText(img, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


model = YOLO("model/yolov8n.pt")
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
threshold = 0.7

def track(frame, counter=0, track_ids=[]):
    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > threshold and class_id == 0:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            if track_id not in track_ids:
                track_ids.append(int(track_id))
                print(track_ids)
                counter += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]),
                          3)
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (colors[track_id % len(colors)]))

        frame = cv2.line(frame, (400, 30), (600, 200), (255, 0, 0), 3)
    return frame, counter


class Tracker:
    def __init__(self):
        self.model = YOLO("model/yolov8n.pt")
        self.tracker = Tracker()
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
        self.threshold = 0.7
        self.counter = 0
        self.track_ids = np.array([])

    def detect_persons(self, frame):
        results = self.model(frame)
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > self.threshold and class_id == 0:
                    detections.append([x1, y1, x2, y2, score])

            self.tracker.update(frame, detections)

            for track in self.tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                if track_id not in self.track_ids:
                    self.counter += 1
                    np.append(self.track_ids, int(track_id))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (self.colors[track_id % len(self.colors)]),
                              3)
                cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (self.colors[track_id % len(self.colors)]))

            frame = cv2.line(frame, (400, 30), (600, 200), (255, 0, 0), 3)
        return frame, self.counter
