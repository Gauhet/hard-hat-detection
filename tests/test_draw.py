import numpy as np
from PIL import Image

from hardhat_detector.draw import COLOURS, draw_boxes


def test_boxes_are_drawn_in_their_class_colour_without_changing_the_input():
    image = Image.new("RGB", (200, 200), "black")
    boxes = np.array([[20, 60, 80, 120], [120, 60, 180, 120]], dtype=float)
    out = draw_boxes(image, boxes, classes=np.array([0, 1]), scores=np.array([0.9, 0.6]))
    assert out.size == image.size and image.getpixel((20, 100)) == (0, 0, 0)
    assert out.getpixel((20, 100)) == COLOURS["helmet"]
    assert out.getpixel((120, 100)) == COLOURS["head"]


def test_drawing_nothing_returns_an_unchanged_copy():
    image = Image.new("RGB", (50, 50), "gray")
    out = draw_boxes(image, np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    assert out.tobytes() == image.tobytes()


def test_labels_can_be_turned_off_for_crowded_scenes():
    image = Image.new("RGB", (200, 200), "black")
    boxes = np.array([[50, 80, 150, 150]], dtype=float)
    labelled = draw_boxes(image, boxes, np.array([1]), np.array([0.9]))
    plain = draw_boxes(image, boxes, np.array([1]), np.array([0.9]), labels=False)
    # The label tag sits just above the box; without labels that area stays black.
    assert labelled.getpixel((60, 70)) == COLOURS["head"]
    assert plain.getpixel((60, 70)) == (0, 0, 0)
