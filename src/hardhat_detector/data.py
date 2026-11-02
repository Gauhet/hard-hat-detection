"""Dataset preparation: download two hard hat datasets, convert both to YOLO format, remove
near-duplicate photos across them, and split the result into train, validation and two test sets.

Sources (both on Hugging Face):
- Hard Hat Workers (public domain): 7,063 photos, mostly wide shots of construction sites.
- Roboflow "Hard Hats" (CC BY 4.0): 19,745 photos, including many close-ups, selfies and people
  in masks, the kind of bare-head photo the first dataset has almost none of.

The Roboflow set already contains most of the Hard Hat Workers photos (resized), and many of
its photos are frames from the same video. So photos are grouped, by near-duplicate hash and by
source video, and each group goes to a single split. Test sets take priority, so no test photo
has a copy, or a neighbouring video frame, in training or validation.
"""

import json
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hardhat_detector.dedupe import group_duplicates, hash_all

# "helmet" is a head wearing a hard hat, "head" is a bare head. Hard Hat Workers also has a
# "person" class, but only a few hundred people are boxed across 7,000 photos full of workers,
# so the labels are too incomplete to learn from and are dropped (as are three boxes labelled
# "others"). Roboflow's "hardhat" and "no-hardhat" boxes are drawn the same way as these.
CLASSES = ("helmet", "head")
ROBOFLOW_NAMES = {"hardhat": "helmet", "no-hardhat": "head"}

HHW_URL = "https://huggingface.co/datasets/hf-vision/hardhat/resolve/main/data/{file}"
HHW_FILES = {
    "train": "train-00000-of-00001-3e0d95da2811afee.parquet",
    "test": "test-00000-of-00001-9fc2e7e45a21ae43.parquet",
}
ROBOFLOW_URL = "https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/{split}.zip"
ROBOFLOW_SPLITS = ("train", "valid", "test")

# Roboflow keeps each photo's original file name; frames cut from a video are named like
# "IMG_1234_mp4-17" (video name, then frame number).
VIDEO_FRAME = re.compile(r"^(.*(?:mp4|mov|avi))-\d+$", re.IGNORECASE)

# Final splits. "test" is the official Hard Hat Workers test set, unchanged so results stay
# comparable; "test_roboflow" is what is left of Roboflow's test set after removing copies.
SPLITS = ("train", "val", "test", "test_roboflow")


def to_yolo_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Convert a COCO box (left, top, box width, box height) in pixels into YOLO's
    (centre x, centre y, width, height) as fractions of the image size.

    Boxes are clipped to the image first; ones with no area left are dropped (None).
    """
    left, top, w, h = box
    x1, y1 = max(left, 0.0), max(top, 0.0)
    x2, y2 = min(left + w, width), min(top + h, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def label_lines(boxes: list[list[float]], categories: list[str], width: int, height: int) -> list[str]:
    """One "class cx cy w h" line per helmet or head, the format YOLO label files use."""
    lines = []
    for box, category in zip(boxes, categories):
        if category not in CLASSES:
            continue
        yolo_box = to_yolo_box(box, width, height)
        if yolo_box is not None:
            lines.append(f"{CLASSES.index(category)} " + " ".join(f"{v:.6f}" for v in yolo_box))
    return lines


def write_sample(image_bytes: bytes, lines: list[str], folder: Path, name: str) -> None:
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    (folder / "images" / f"{name}.jpg").write_bytes(image_bytes)
    (folder / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n" if lines else "")


def download_hard_hat_workers(sources: Path) -> None:
    for split, file in HHW_FILES.items():
        parquet = sources / file
        print(f"downloading Hard Hat Workers {split} split...")
        urllib.request.urlretrieve(HHW_URL.format(file=file), parquet)
        for row in pq.read_table(parquet).to_pylist():
            objects = row["objects"]
            lines = label_lines(objects["bbox"], objects["category"], row["width"], row["height"])
            write_sample(row["image"]["bytes"], lines, sources / "hhw" / split, Path(row["image"]["path"]).stem)
        parquet.unlink()


def source_video(original_name: str) -> str | None:
    """The video a Roboflow photo was cut from, or None for a still photo."""
    stem = original_name.split(".rf.")[0].removesuffix("_jpg").removesuffix("_png")
    match = VIDEO_FRAME.match(stem)
    return match[1] if match else None


def download_roboflow(sources: Path) -> None:
    for split in ROBOFLOW_SPLITS:
        archive = sources / f"roboflow_{split}.zip"
        if not archive.exists():
            print(f"downloading Roboflow Hard Hats {split} split...")
            urllib.request.urlretrieve(ROBOFLOW_URL.format(split=split), archive)
        originals = {}
        with zipfile.ZipFile(archive) as zf:
            coco = json.loads(zf.read("_annotations.coco.json"))
            names = {c["id"]: ROBOFLOW_NAMES.get(c["name"], c["name"]) for c in coco["categories"]}
            boxes: dict[int, list] = {}
            for a in coco["annotations"]:
                boxes.setdefault(a["image_id"], []).append(a)
            for k, image in enumerate(coco["images"]):
                anns = boxes.get(image["id"], [])
                lines = label_lines([a["bbox"] for a in anns], [names[a["category_id"]] for a in anns],
                                    image["width"], image["height"])
                name = f"rf_{split}_{k:05d}"
                write_sample(zf.read(image["file_name"]), lines, sources / "roboflow" / split, name)
                originals[name] = image["file_name"]
        (sources / "roboflow" / split / "original_names.json").write_text(json.dumps(originals, indent=0))
        archive.unlink()


def assign_splits(
    origins: list[str], groups: np.ndarray, val_fraction: float, seed: int, max_test_group: int = 10
) -> list[str | None]:
    """Choose a split for every photo, given where it came from ("hhw_test", "rf_test", or
    anything else for training material) and its near-duplicate group.

    - A group with a Hard Hat Workers test photo: those photos go to "test", the rest are dropped,
      so the original test set stays exactly as published and nothing like it is trained on.
    - Otherwise, a small group with a Roboflow test photo: one of its test photos goes to
      "test_roboflow" and the rest of the group is dropped.
    - Every other group goes whole to "val" or "train" (`val_fraction` of groups to val). That
      includes large groups, which are runs of frames from one fixed camera: keeping them out
      of testing means no camera view is both trained on and tested on.
    Dropped photos get None.
    """
    members: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(int(g), []).append(i)
    order = sorted(members)
    random.Random(seed).shuffle(order)
    val_groups = set(order[: round(len(order) * val_fraction)])

    split: list[str | None] = [None] * len(origins)
    for g, idx in members.items():
        hhw_test = [i for i in idx if origins[i] == "hhw_test"]
        rf_test = [i for i in idx if origins[i] == "rf_test"]
        if hhw_test:
            for i in hhw_test:
                split[i] = "test"
        elif rf_test and len(idx) <= max_test_group:
            split[rf_test[0]] = "test_roboflow"
        else:
            for i in idx:
                split[i] = "val" if g in val_groups else "train"
    return split


def write_dataset_yaml(data_dir: Path) -> Path:
    path = data_dir / "hardhat.yaml"
    root = data_dir.resolve().as_posix()
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    # Absolute paths: Ultralytics only resolves train/val/test against `path`, not extra splits.
    folders = "".join(f"{s}: {root}/images/{s}\n" for s in SPLITS)
    path.write_text(f"path: {root}\n{folders}names:\n{names}\n")
    return path


def prepare_dataset(data_dir: str | Path = "data", val_fraction: float = 0.1, seed: int = 42) -> Path:
    """Download, convert, de-duplicate and split the data on first use, and return the path of
    the YOLO dataset file."""
    data_dir = Path(data_dir)
    yaml_path = data_dir / "hardhat.yaml"
    if yaml_path.exists():
        return yaml_path

    sources = data_dir / "sources"
    if not (sources / "hhw").exists():
        download_hard_hat_workers(sources)
    if not (sources / "roboflow").exists():
        download_roboflow(sources)

    images, origins, videos = [], [], []
    for dataset, splits in (("hhw", HHW_FILES), ("roboflow", ROBOFLOW_SPLITS)):
        for split in splits:
            found = sorted((sources / dataset / split / "images").glob("*.jpg"))
            images += found
            origins += [f"{'hhw' if dataset == 'hhw' else 'rf'}_{split}"] * len(found)
            names_file = sources / dataset / split / "original_names.json"
            originals = json.loads(names_file.read_text()) if names_file.exists() else {}
            videos += [source_video(originals[p.stem]) if p.stem in originals else None for p in found]

    print(f"hashing {len(images):,} photos to find duplicates...")
    groups = group_duplicates(hash_all(images), sources=videos)
    splits = assign_splits(origins, groups, val_fraction, seed)

    for split in SPLITS:
        for kind in ("images", "labels"):
            (data_dir / kind / split).mkdir(parents=True, exist_ok=True)
    for image, split in zip(images, splits):
        if split is None:
            continue
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        shutil.copy(image, data_dir / "images" / split / image.name)
        shutil.copy(label, data_dir / "labels" / split / label.name)

    counts = {s: splits.count(s) for s in SPLITS}
    counts["dropped duplicates"] = splits.count(None)
    (data_dir / "split_counts.json").write_text(json.dumps(counts, indent=2))
    print(counts)
    return write_dataset_yaml(data_dir)


if __name__ == "__main__":
    print(prepare_dataset())
