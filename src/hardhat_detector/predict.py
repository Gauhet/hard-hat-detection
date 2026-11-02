"""Find heads with and without hard hats in your own photos or videos, and save annotated copies.

    python -m hardhat_detector.predict site.jpg [more photos or videos...]
    python -m hardhat_detector.predict clip.mp4 --model checkpoints/yolov8n.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from hardhat_detector.data import CLASSES
from hardhat_detector.draw import draw_boxes, draw_detections

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def count(result) -> dict[str, int]:
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return {name: int((classes == i).sum()) for i, name in enumerate(CLASSES)}


def annotate_image(model: YOLO, path: str | Path, conf: float = 0.25) -> tuple[Image.Image, dict[str, int]]:
    image = Image.open(path).convert("RGB")
    result = model.predict(image, conf=conf, verbose=False)[0]
    return draw_detections(image, result), count(result)


def annotate_video(
    model: YOLO, path: str | Path, out_path: str | Path, conf: float = 0.25, max_frames: int | None = None
) -> dict:
    """Run the detector on every frame and write the annotated video to `out_path`.

    A .webm output (VP8) plays in browsers; .mp4 uses the widely supported mp4v codec.
    `max_frames` stops early, for keeping demo uploads quick.
    Returns how many frames showed at least one bare head, and the most heads seen at once.
    """
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise ValueError(f"could not open video {path}")
    fps = reader.get(cv2.CAP_PROP_FPS) or 25
    size = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = "VP80" if Path(out_path).suffix == ".webm" else "mp4v"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size)

    frames = frames_with_bare_head = most_heads = 0
    while max_frames is None or frames < max_frames:
        ok, frame = reader.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = model.predict(image, conf=conf, verbose=False)[0].boxes
        classes = boxes.cls.cpu().numpy()
        annotated = draw_boxes(image, boxes.xyxy.cpu().numpy(), classes, boxes.conf.cpu().numpy())
        writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
        frames += 1
        frames_with_bare_head += int((classes == CLASSES.index("head")).any())
        most_heads = max(most_heads, len(classes))
    reader.release()
    writer.release()
    return {"frames": frames, "frames_with_bare_head": frames_with_bare_head, "most_heads_in_a_frame": most_heads}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--model", default="checkpoints/yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out-dir", default="predictions")
    args = parser.parse_args()

    model = YOLO(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.inputs):
        if path.suffix.lower() in VIDEO_SUFFIXES:
            out_path = out_dir / f"{path.stem}_detected.mp4"
            stats = annotate_video(model, path, out_path, args.conf)
            print(f"{path}: bare head in {stats['frames_with_bare_head']} of {stats['frames']} frames -> {out_path}")
        else:
            image, counts = annotate_image(model, path, args.conf)
            out_path = out_dir / f"{path.stem}_detected.jpg"
            image.save(out_path)
            print(f"{path}: {counts['helmet']} with hard hat, {counts['head']} without -> {out_path}")


if __name__ == "__main__":
    main()
