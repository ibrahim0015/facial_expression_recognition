from pathlib import Path

import cv2
import numpy as np


def load_haar_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar face detector.")
    return detector


def detect_faces(frame_bgr: np.ndarray, detector=None):
    detector = detector or load_haar_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)


def draw_prediction(frame_bgr: np.ndarray, box, label: str, confidence: float):
    x, y, w, h = box
    text = f"{label}: {confidence:.0%}"
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (20, 180, 90), 2)
    cv2.rectangle(frame_bgr, (x, max(0, y - 28)), (x + w, y), (20, 180, 90), -1)
    cv2.putText(
        frame_bgr,
        text,
        (x + 6, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
