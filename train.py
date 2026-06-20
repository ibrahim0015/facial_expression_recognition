import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import make_image_folder
from src.inference import get_device
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch CNN for facial expression recognition.")
    parser.add_argument("--train-dir", default="data/fer2013/train")
    parser.add_argument("--val-dir", default="data/fer2013/test")
    parser.add_argument("--model-name", choices=["custom_cnn", "resnet18", "mobilenet_v3_small"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="models/emotion_cnn_best.pt")
    parser.add_argument("--no-pretrained", action="store_true", help="Train selected architecture without ImageNet weights.")
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights.")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    all_preds = []
    all_targets = []

    loop = tqdm(loader, leave=False)
    for images, targets in loop:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": accuracy_score(all_targets, all_preds),
    }


def main():
    args = parse_args()
    device = get_device()

    train_dataset = make_image_folder(args.train_dir, args.image_size, train=True, model_name=args.model_name)
    val_dataset = make_image_folder(args.val_dir, args.image_size, train=False, model_name=args.model_name)
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        num_classes=len(class_names),
        model_name=args.model_name,
        pretrained=not args.no_pretrained and args.model_name != "custom_cnn",
    ).to(device)

    class_weight_tensor = None
    if args.class_weights:
        counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(class_names)).float()
        weights = counts.sum() / (counts.clamp_min(1) * len(class_names))
        class_weight_tensor = weights.to(device)
        print(f"Class weights: {[round(weight, 3) for weight in weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model_name} | Image size: {args.image_size} | Pretrained: {not args.no_pretrained and args.model_name != 'custom_cnn'}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "model_name": args.model_name,
                    "val_accuracy": best_val_acc,
                },
                output_path,
            )
            print(f"Saved new best checkpoint: {output_path} ({best_val_acc:.4f})")


if __name__ == "__main__":
    main()
