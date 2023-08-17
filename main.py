from ultralytics import YOLO
import cv2
import random
import math
import logging
import numpy as np
from tracker import Tracker


model = YOLO("model/yolov8n.pt")
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
threshold = 0.7


def track(frame, counter=0, track_ids=[]):
    '''
    Detects the poeple in the frame and tracks them using deep_sort tracker.
    '''
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
                counter += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]),
                          3)
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (colors[track_id % len(colors)]))

        frame = cv2.line(frame, (400, 30), (600, 200), (255, 0, 0), 3)
    return frame, counter, track_ids


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
