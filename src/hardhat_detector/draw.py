"""Draw detections on a photo: green boxes for hard hats, red for bare heads."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hardhat_detector.data import CLASSES

COLOURS = {"helmet": (34, 197, 94), "head": (239, 68, 68)}


def draw_boxes(
    image: Image.Image, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, labels: bool = True
) -> Image.Image:
    """Return a copy of `image` with each (x1, y1, x2, y2) box outlined and, if `labels`,
    tagged with its class and score (crowded scenes are easier to read without them)."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    line = max(2, round(max(image.size) / 300))
    font = ImageFont.load_default(size=max(11, round(max(image.size) / 45)))
    # Draw the least confident first so the most confident labels end up on top.
    for i in np.argsort(scores):
        colour = COLOURS[CLASSES[int(classes[i])]]
        x1, y1, x2, y2 = boxes[i].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=line)
        if not labels:
            continue
        text = f"{CLASSES[int(classes[i])]} {scores[i]:.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        label_top = max(0, y1 - (bottom - top) - 2)
        draw.rectangle([x1, label_top, x1 + (right - left) + 4, label_top + (bottom - top) + 2], fill=colour)
        draw.text((x1 + 2, label_top - (top - y1)), text, fill="white", font=font)
    return image


def draw_detections(image: Image.Image, result, labels: bool = True) -> Image.Image:
    """Draw one Ultralytics result on the photo it came from."""
    detections = result.boxes
    return draw_boxes(
        image, detections.xyxy.cpu().numpy(), detections.cls.cpu().numpy(), detections.conf.cpu().numpy(), labels
    )
