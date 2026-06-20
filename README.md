# Real-Time Facial Expression Recognition

PyTorch + Streamlit project for facial expression classification from uploaded images, camera snapshots, and uploaded videos.

This app predicts facial expression classes such as:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

Treat the output as facial expression classification, not a guaranteed reading of a person's true emotion.

## Dataset

Start with FER-2013 from Kaggle. Put it in this folder style:

```text
data/fer2013/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

If your Kaggle download uses `Training` and `Testing`, rename them to `train` and `test`, or pass the paths manually to the training script.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Transfer Learning Model

```bash
python train.py --train-dir data/fer2013/train --val-dir data/fer2013/test --epochs 35 --batch-size 32 --class-weights
```

The best checkpoint is saved to:

```text
models/emotion_cnn_best.pt
```

By default, training now uses:

```text
model: pretrained ResNet18
image size: 224x224
optimizer: AdamW
augmentation: flip, rotation, affine, color jitter, random erasing
```

To train the old custom CNN baseline instead:

```bash
python train.py --model-name custom_cnn --image-size 48 --lr 1e-3 --batch-size 64 --epochs 25
```

## Colab Training

After uploading this project and your dataset to Colab, run:

```python
!pip install torch torchvision scikit-learn tqdm matplotlib seaborn

!python "/content/CNN project/train.py" \
  --train-dir "/content/your_train_folder" \
  --val-dir "/content/your_test_folder" \
  --epochs 35 \
  --batch-size 32 \
  --class-weights \
  --output "/content/emotion_cnn_best.pt"
```

Then download:

```python
from google.colab import files
files.download("/content/emotion_cnn_best.pt")
```

## Evaluate

```bash
python evaluate.py --checkpoint models/emotion_cnn_best.pt --data-dir data/fer2013/test
```

Outputs are saved under:

```text
outputs/
```

## Run Streamlit

```bash
streamlit run app.py
```

The app supports:

- Image upload
- Camera snapshot
- Uploaded video annotation
