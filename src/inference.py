from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data import get_eval_transforms
from src.model import build_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: str | Path, device: torch.device | None = None):
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 48)
    model_name = checkpoint.get("model_name", "custom_cnn")

    model = build_model(num_classes=len(class_names), model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_eval_transforms(image_size, model_name)
    return model, class_names, transform, device


def predict_pil_image(model, image: Image.Image, transform, class_names, device):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    best_idx = int(np.argmax(probabilities))
    return {
        "label": class_names[best_idx],
        "confidence": float(probabilities[best_idx]),
        "probabilities": {name: float(probabilities[idx]) for idx, name in enumerate(class_names)},
    }


def crop_bgr(frame_bgr: np.ndarray, box):
    x, y, w, h = box
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    return frame_bgr[y1:y2, x1:x2]


def bgr_to_pil_gray(face_bgr: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray)


def bgr_to_pil_rgb(face_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
