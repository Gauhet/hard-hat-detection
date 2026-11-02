import pytest

from hardhat_detector.data import label_lines, split_names, to_yolo_box, write_dataset_yaml


def test_coco_box_becomes_normalised_centre_box():
    # A 20x10 box whose top-left corner is at (40, 30), in a 200x100 photo.
    assert to_yolo_box([40, 30, 20, 10], 200, 100) == pytest.approx((0.25, 0.35, 0.1, 0.1))


def test_boxes_are_clipped_to_the_photo_and_empty_ones_dropped():
    assert to_yolo_box([-10, 0, 20, 10], 100, 100) == pytest.approx((0.05, 0.05, 0.1, 0.1))
    assert to_yolo_box([120, 0, 10, 10], 100, 100) is None


def test_label_lines_keep_only_helmet_and_head():
    objects = {
        "bbox": [[0, 0, 10, 10], [10, 10, 10, 10], [0, 0, 50, 90], [5, 5, 5, 5]],
        "category": ["helmet", "head", "person", "others"],
    }
    lines = label_lines(objects, 100, 100)
    assert [line.split()[0] for line in lines] == ["0", "1"]
    assert all(len(line.split()) == 5 for line in lines)


def test_split_is_disjoint_complete_and_repeatable():
    names = [f"{i:06d}" for i in range(100)]
    train, val = split_names(names, 0.1, seed=1)
    assert len(val) == 10 and set(train).isdisjoint(val)
    assert sorted(train + val) == names
    assert split_names(list(reversed(names)), 0.1, seed=1) == (train, val)


def test_dataset_yaml_lists_splits_and_classes(tmp_path):
    text = write_dataset_yaml(tmp_path).read_text()
    for line in ("train: images/train", "val: images/val", "test: images/test", "0: helmet", "1: head"):
        assert line in text
