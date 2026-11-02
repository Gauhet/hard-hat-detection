import numpy as np
import pytest

from hardhat_detector.data import assign_splits, label_lines, source_video, to_yolo_box, write_dataset_yaml


def test_coco_box_becomes_normalised_centre_box():
    # A 20x10 box whose top-left corner is at (40, 30), in a 200x100 photo.
    assert to_yolo_box([40, 30, 20, 10], 200, 100) == pytest.approx((0.25, 0.35, 0.1, 0.1))


def test_boxes_are_clipped_to_the_photo_and_empty_ones_dropped():
    assert to_yolo_box([-10, 0, 20, 10], 100, 100) == pytest.approx((0.05, 0.05, 0.1, 0.1))
    assert to_yolo_box([120, 0, 10, 10], 100, 100) is None


def test_label_lines_keep_only_helmet_and_head():
    boxes = [[0, 0, 10, 10], [10, 10, 10, 10], [0, 0, 50, 90], [5, 5, 5, 5]]
    lines = label_lines(boxes, ["helmet", "head", "person", "others"], 100, 100)
    assert [line.split()[0] for line in lines] == ["0", "1"]
    assert all(len(line.split()) == 5 for line in lines)


def test_video_frames_are_traced_to_their_video():
    assert source_video("IMG_1234_mp4-17_jpg.rf.abc123.jpg") == "IMG_1234_mp4"
    assert source_video("Mask3_MOV-2_jpg.rf.abc123.jpg") == "Mask3_MOV"
    assert source_video("ppe_0012_jpg.rf.abc123.jpg") is None


def test_test_photos_never_share_a_group_with_training_photos():
    origins = ["hhw_test", "rf_train", "rf_test", "rf_train", "hhw_train", "rf_valid"]
    groups = np.array([0, 0, 1, 1, 2, 3])
    splits = assign_splits(origins, groups, val_fraction=0.0, seed=0)
    # Group 0 holds an original test photo: it stays in test and its copy is dropped.
    # Group 1 holds a Roboflow test photo: it goes to test_roboflow and its copy is dropped.
    assert splits == ["test", None, "test_roboflow", None, "train", "train"]


def test_large_groups_go_to_training_whole_even_with_a_test_frame():
    # A long run of frames from one camera: keeping it out of testing keeps every frame useful.
    origins = ["rf_test"] + ["rf_train"] * 11
    splits = assign_splits(origins, np.zeros(12, dtype=int), val_fraction=0.0, seed=0, max_test_group=10)
    assert splits == ["train"] * 12


def test_validation_is_chosen_by_group_and_repeatable():
    origins = ["rf_train"] * 200
    groups = np.arange(200) // 2  # 100 groups of two
    splits = assign_splits(origins, groups, val_fraction=0.2, seed=3)
    assert splits.count("val") == 40
    assert all(splits[i] == splits[i + 1] for i in range(0, 200, 2))
    assert assign_splits(origins, groups, val_fraction=0.2, seed=3) == splits


def test_dataset_yaml_lists_splits_and_classes(tmp_path):
    lines = write_dataset_yaml(tmp_path).read_text().splitlines()
    root = tmp_path.resolve().as_posix()
    for split in ("train", "val", "test", "test_roboflow"):
        assert f"{split}: {root}/images/{split}" in lines
    assert "  0: helmet" in lines and "  1: head" in lines
