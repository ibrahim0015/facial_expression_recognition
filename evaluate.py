import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import make_image_folder
from src.inference import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an emotion recognition checkpoint.")
    parser.add_argument("--checkpoint", default="models/emotion_cnn_best.pt")
    parser.add_argument("--data-dir", default="data/fer2013/test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    model, class_names, _transform, device = load_checkpoint(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = checkpoint.get("image_size", 48)
    model_name = checkpoint.get("model_name", "custom_cnn")

    dataset = make_image_folder(args.data_dir, image_size, train=False, model_name=model_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true = []
    y_pred = []
    model.eval()

    for images, targets in tqdm(loader):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(targets.tolist())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
