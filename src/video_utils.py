from pathlib import Path

import cv2

from src.face_detection import detect_faces, draw_prediction, load_haar_detector
from src.inference import bgr_to_pil_gray, crop_bgr, predict_pil_image


def annotate_video(input_path, output_path, model, transform, class_names, device, every_n_frames: int = 3):
    input_path = str(input_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = load_haar_detector()
    cached_predictions = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames == 0:
            cached_predictions = []
            faces = detect_faces(frame, detector)
            for box in faces:
                face = crop_bgr(frame, box)
                if face.size == 0:
                    continue
                result = predict_pil_image(model, bgr_to_pil_gray(face), transform, class_names, device)
                cached_predictions.append((box, result["label"], result["confidence"]))

        for box, label, confidence in cached_predictions:
            frame = draw_prediction(frame, box, label, confidence)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return {"frames": frame_idx, "total_frames": total_frames, "output_path": output_path}
