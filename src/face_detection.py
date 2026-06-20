from pathlib import Path

import cv2
import numpy as np


def load_haar_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar face detector.")
    return detector


def detect_faces(
    frame_bgr: np.ndarray,
    detector=None,
    max_faces: int | None = 3,
    min_face_ratio: float = 0.08,
    min_confidence: float = 2.0,
):
    detector = detector or load_haar_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    height, width = gray.shape[:2]
    min_face_size = max(50, int(min(height, width) * min_face_ratio))

    try:
        faces, _reject_levels, level_weights = detector.detectMultiScale3(
            gray,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(min_face_size, min_face_size),
            outputRejectLevels=True,
        )
        weighted_faces = [
            (tuple(face), float(weight))
            for face, weight in zip(faces, level_weights)
            if weight >= min_confidence
        ]
    except cv2.error:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(min_face_size, min_face_size),
        )
        weighted_faces = [(tuple(face), 0.0) for face in faces]

    filtered_faces = []
    image_area = width * height
    for box, weight in weighted_faces:
        x, y, w, h = box
        aspect_ratio = w / max(1, h)
        area_ratio = (w * h) / max(1, image_area)

        if not 0.75 <= aspect_ratio <= 1.35:
            continue
        if area_ratio < 0.004:
            continue

        filtered_faces.append((box, weight))

    filtered_faces.sort(key=lambda item: (item[0][2] * item[0][3], item[1]), reverse=True)
    boxes = [box for box, _weight in filtered_faces]
    return boxes[:max_faces] if max_faces else boxes


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
