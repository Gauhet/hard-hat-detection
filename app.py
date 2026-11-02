"""A small web demo: upload a construction-site photo or video clip and see who is and isn't
wearing a hard hat.

    pip install -e ".[demo]"
    python app.py
"""

import random
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO

from hardhat_detector.data import prepare_dataset
from hardhat_detector.draw import draw_detections
from hardhat_detector.predict import annotate_video, count

ROOT = Path(__file__).parent
MODELS = {path.stem: YOLO(path) for path in sorted((ROOT / "checkpoints").glob("yolov8*.pt"))}
DEFAULT_MODEL = "yolov8s" if "yolov8s" in MODELS else next(iter(MODELS))
prepare_dataset(ROOT / "data")
TEST_PHOTOS = sorted((ROOT / "data" / "images" / "test").glob("*.jpg"))
MAX_VIDEO_SECONDS = 20


def summary(counts: dict[str, int]) -> str:
    if counts["helmet"] + counts["head"] == 0:
        return "No heads found."
    return f"**{counts['helmet']}** wearing a hard hat, **{counts['head']}** without."


def detect_photo(image: Image.Image | None, model_name: str, conf: float) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, ""
    result = MODELS[model_name].predict(image, conf=conf, verbose=False)[0]
    return draw_detections(image.convert("RGB"), result), summary(count(result))


def random_photo(model_name: str, conf: float) -> tuple[Image.Image, Image.Image, str]:
    image = Image.open(random.choice(TEST_PHOTOS)).convert("RGB")
    return (image, *detect_photo(image, model_name, conf))


def detect_video(video: str | None, model_name: str, conf: float) -> tuple[str | None, str]:
    if video is None:
        return None, ""
    out_path = Path(tempfile.mkdtemp()) / "detected.webm"
    fps = 30  # an upper bound for the frame limit; most phone clips are 30 fps or less
    stats = annotate_video(MODELS[model_name], video, out_path, conf, max_frames=MAX_VIDEO_SECONDS * fps)
    share = stats["frames_with_bare_head"] / max(stats["frames"], 1)
    return str(out_path), (
        f"Processed {stats['frames']} frames. A bare head was visible in "
        f"**{stats['frames_with_bare_head']}** of them ({share:.0%})."
    )


with gr.Blocks(title="Hard Hat Detection") as demo:
    gr.Markdown(
        "# Hard Hat Detection\n"
        "YOLOv8 fine-tuned to find construction workers' heads and say whether each one is wearing a hard hat "
        "(green) or not (red)."
    )
    with gr.Row():
        model_name = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")

    with gr.Tab("Photo"):
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Photo", height=420)
                with gr.Row():
                    random_button = gr.Button("Random test photo", variant="primary")
                    detect_button = gr.Button("Detect")
            with gr.Column():
                photo_out = gr.Image(label="Detections", height=420)
                photo_summary = gr.Markdown()
        random_button.click(random_photo, inputs=[model_name, conf], outputs=[photo, photo_out, photo_summary])
        detect_button.click(detect_photo, inputs=[photo, model_name, conf], outputs=[photo_out, photo_summary])
        photo.upload(detect_photo, inputs=[photo, model_name, conf], outputs=[photo_out, photo_summary])

    with gr.Tab("Video"):
        gr.Markdown(f"Upload a short clip; only the first {MAX_VIDEO_SECONDS} seconds are processed.")
        with gr.Row():
            video = gr.Video(label="Video", height=420)
            with gr.Column():
                video_out = gr.Video(label="Detections", height=420)
                video_summary = gr.Markdown()
        gr.Button("Detect", variant="primary").click(
            detect_video, inputs=[video, model_name, conf], outputs=[video_out, video_summary]
        )


if __name__ == "__main__":
    demo.launch()
