# Facial Expression Recognition with PyTorch and Streamlit

A computer vision web app that classifies facial expressions from images, videos, camera snapshots, and live webcam input.

The project uses a PyTorch transfer-learning model for expression classification, OpenCV for face detection, and Streamlit for the interactive web interface.

> This project performs facial expression classification. It should not be treated as a reliable measurement of a person's true internal emotion.

## Demo

Add your screenshots here after deploying or running the app locally.

```text
assets/demo-image-upload.png
assets/demo-webcam.png
assets/demo-video.png
```

Suggested screenshots:

- Image upload with detected face and prediction
- Live webcam mode
- Confidence score chart
- Confusion matrix from evaluation

## Features

- Upload an image and classify the detected facial expression
- Take a camera snapshot directly inside the app
- Run live webcam inference
- Upload a video and generate an annotated output video
- Display prediction confidence scores
- Evaluate the trained model with a classification report and confusion matrix
- Train either a custom CNN baseline or a transfer-learning model

## Expression Classes

The model predicts one of seven expression categories:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

## Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- scikit-learn
- Matplotlib and Seaborn

## How It Works

```text
Input image/video/webcam frame
        |
        v
OpenCV face detection
        |
        v
Face crop extraction
        |
        v
Resize and normalize image
        |
        v
PyTorch emotion classifier
        |
        v
Predicted expression + confidence scores
```

## Model

The main model uses transfer learning with a pretrained ResNet18 backbone. The final classification layer is replaced for seven facial expression classes.

The project also includes a smaller custom CNN baseline for comparison.

Default transfer-learning setup:

```text
Architecture: ResNet18
Pretraining: ImageNet
Input size: 224x224
Optimizer: AdamW
Loss: CrossEntropyLoss
Augmentation: horizontal flip, rotation, affine transform, color jitter, random erasing
```

## Project Structure

```text
.
├── app.py                  # Streamlit web app
├── train.py                # Model training script
├── evaluate.py             # Evaluation and confusion matrix script
├── requirements.txt        # Python dependencies
├── models/
│   └── emotion_cnn_best.pt # Trained checkpoint
├── src/
│   ├── data.py             # Dataset loading and transforms
│   ├── face_detection.py   # OpenCV face detection helpers
│   ├── inference.py        # Checkpoint loading and prediction helpers
│   ├── model.py            # Custom CNN and transfer-learning models
│   └── video_utils.py      # Video annotation pipeline
└── data/
    └── README.md           # Dataset instructions
```

## Dataset

This project can be trained on FER-2013 style facial expression datasets.

Expected folder format:

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

The dataset is not included in this repository because of size and licensing. Download it separately and place it inside the `data/` folder.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the default transfer-learning model:

```bash
python train.py --train-dir data/fer2013/train --val-dir data/fer2013/test --epochs 35 --batch-size 32 --class-weights
```

The best checkpoint is saved to:

```text
models/emotion_cnn_best.pt
```

Train the custom CNN baseline:

```bash
python train.py --model-name custom_cnn --image-size 48 --lr 1e-3 --batch-size 64 --epochs 25
```

## Evaluation

Evaluate the trained model:

```bash
python evaluate.py --checkpoint models/emotion_cnn_best.pt --data-dir data/fer2013/test
```

Evaluation outputs are saved in:

```text
outputs/
```

The evaluation script generates:

- Classification report
- Confusion matrix

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

The app expects the trained model checkpoint at:

```text
models/emotion_cnn_best.pt
```

## Deployment

The app can be deployed for free on Streamlit Community Cloud.

Basic deployment steps:

1. Push the project to GitHub.
2. Make sure `requirements.txt` is in the repository root.
3. Make sure `models/emotion_cnn_best.pt` is included.
4. Go to Streamlit Community Cloud.
5. Select the GitHub repository.
6. Set the main file path to `app.py`.
7. Deploy.

Do not upload the dataset to GitHub.

## Limitations

- Facial expression recognition is not the same as true emotion understanding.
- FER-style datasets are noisy and often imbalanced.
- Classes such as `fear`, `sad`, and `disgust` can be difficult to classify reliably.
- Face detection quality affects prediction quality.
- Real-world lighting, pose, blur, and camera quality can reduce accuracy.

## Future Improvements

- Train on a larger dataset such as RAF-DB or AffectNet
- Add MediaPipe or RetinaFace for stronger face detection
- Add Grad-CAM visualizations
- Improve video performance with frame skipping and prediction smoothing
- Add model comparison between custom CNN, ResNet18, and MobileNetV3

