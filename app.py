from pathlib import Path
from tempfile import NamedTemporaryFile

import av
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from src.face_detection import detect_faces, draw_prediction, load_haar_detector
from src.inference import bgr_to_pil_gray, crop_bgr, load_checkpoint, predict_pil_image
from src.video_utils import annotate_video


DEFAULT_CHECKPOINT = "models/emotion_cnn_best.pt"
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


st.set_page_config(page_title="Facial Expression Recognition", page_icon=":camera:", layout="wide")


@st.cache_resource
def cached_model(checkpoint_path):
    return load_checkpoint(checkpoint_path)


@st.cache_resource
def cached_detector():
    return load_haar_detector()


def predict_on_bgr_frame(frame_bgr, model, class_names, transform, device, max_faces=1):
    detector = cached_detector()
    faces = detect_faces(frame_bgr, detector, max_faces=max_faces)
    predictions = []

    for box in faces:
        face = crop_bgr(frame_bgr, box)
        if face.size == 0:
            continue
        result = predict_pil_image(model, bgr_to_pil_gray(face), transform, class_names, device)
        predictions.append((box, result))
        frame_bgr = draw_prediction(frame_bgr, box, result["label"], result["confidence"])

    return frame_bgr, predictions


def show_prediction_table(result):
    scores = pd.DataFrame(
        [{"emotion": label, "confidence": confidence} for label, confidence in result["probabilities"].items()]
    ).sort_values("confidence", ascending=False)
    st.bar_chart(scores.set_index("emotion"))
    st.dataframe(scores, hide_index=True, use_container_width=True)


def pil_to_bgr(image: Image.Image):
    import numpy as np

    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


st.title("Facial Expression Recognition")

checkpoint_path = st.sidebar.text_input("Model checkpoint", DEFAULT_CHECKPOINT)
mode = st.sidebar.radio("Input type", ["Image upload", "Camera snapshot", "Live webcam", "Video upload"])
max_faces = st.sidebar.slider("Maximum faces to detect", 1, 10, 3)

if not Path(checkpoint_path).exists():
    st.warning(
        "Train the model first, then place the checkpoint at "
        f"`{checkpoint_path}`. You can still explore the UI."
    )
    st.stop()

model, class_names, transform, device = cached_model(checkpoint_path)

if mode == "Image upload":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded)
        frame = pil_to_bgr(image)
        annotated, predictions = predict_on_bgr_frame(frame, model, class_names, transform, device, max_faces=max_faces)

        left, right = st.columns([2, 1])
        with left:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected faces", use_container_width=True)
        with right:
            if predictions:
                st.subheader("Top face prediction")
                show_prediction_table(predictions[0][1])
            else:
                st.info("No face detected. Try a clearer front-facing image.")

elif mode == "Camera snapshot":
    camera_image = st.camera_input("Take a snapshot")
    if camera_image:
        image = Image.open(camera_image)
        frame = pil_to_bgr(image)
        annotated, predictions = predict_on_bgr_frame(frame, model, class_names, transform, device, max_faces=max_faces)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Camera prediction", use_container_width=True)
        if predictions:
            show_prediction_table(predictions[0][1])
        else:
            st.info("No face detected. Try better lighting and face the camera.")

elif mode == "Live webcam":
    st.caption("Start the stream, allow camera access, and face the camera.")

    class EmotionVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            frame_bgr = frame.to_ndarray(format="bgr24")
            annotated, _predictions = predict_on_bgr_frame(
                frame_bgr,
                model,
                class_names,
                transform,
                device,
                max_faces=max_faces,
            )
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="emotion-live-webcam",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    every_n_frames = st.slider("Run detection every N frames", 1, 10, 1)

    if uploaded_video and st.button("Annotate video"):
        with NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as input_tmp:
            input_tmp.write(uploaded_video.read())
            input_path = input_tmp.name

        output_path = "outputs/annotated_video.mp4"
        with st.spinner("Annotating video..."):
            result = annotate_video(
                input_path,
                output_path,
                model,
                transform,
                class_names,
                device,
                every_n_frames=every_n_frames,
            )

        st.success(f"Processed {result['frames']} frames.")
        st.video(output_path)
