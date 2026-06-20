from pathlib import Path

from torchvision import datasets, transforms


FER_MEAN = (0.5,)
FER_STD = (0.5,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int, model_name: str = "custom_cnn"):
    if model_name == "custom_cnn":
        color_transform = transforms.Grayscale(num_output_channels=1)
        mean = FER_MEAN
        std = FER_STD
    else:
        color_transform = transforms.Lambda(lambda image: image.convert("RGB"))
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    return transforms.Compose(
        [
            color_transform,
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15) if model_name != "custom_cnn" else transforms.Lambda(lambda image: image),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)) if model_name != "custom_cnn" else transforms.Lambda(lambda tensor: tensor),
        ]
    )


def get_eval_transforms(image_size: int, model_name: str = "custom_cnn"):
    if model_name == "custom_cnn":
        color_transform = transforms.Grayscale(num_output_channels=1)
        mean = FER_MEAN
        std = FER_STD
    else:
        color_transform = transforms.Lambda(lambda image: image.convert("RGB"))
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    return transforms.Compose(
        [
            color_transform,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def make_image_folder(data_dir: str | Path, image_size: int, train: bool, model_name: str = "custom_cnn"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    transform = (
        get_train_transforms(image_size, model_name)
        if train
        else get_eval_transforms(image_size, model_name)
    )
    return datasets.ImageFolder(root=str(data_dir), transform=transform)
