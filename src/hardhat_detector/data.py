"""Hard Hat Workers loading: download, conversion to YOLO format, and a fixed train/validation split."""

import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# "helmet" is a head wearing a hard hat, "head" is a bare head. The source also has a
# "person" class, but only a few hundred people are boxed across 7,000 photos full of
# workers, so the labels are too incomplete to learn from and are dropped (as are the
# three boxes labelled "others").
CLASSES = ("helmet", "head")

# The Hugging Face mirror of the dataset, already split into train and test.
SOURCE_URL = "https://huggingface.co/datasets/hf-vision/hardhat/resolve/main/data/{file}"
SOURCE_FILES = {
    "train": "train-00000-of-00001-3e0d95da2811afee.parquet",
    "test": "test-00000-of-00001-9fc2e7e45a21ae43.parquet",
}


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


def label_lines(objects: dict, width: int, height: int) -> list[str]:
    """One "class cx cy w h" line per helmet or head, the format YOLO label files use."""
    lines = []
    for box, category in zip(objects["bbox"], objects["category"]):
        if category not in CLASSES:
            continue
        yolo_box = to_yolo_box(box, width, height)
        if yolo_box is not None:
            lines.append(f"{CLASSES.index(category)} " + " ".join(f"{v:.6f}" for v in yolo_box))
    return lines


def split_names(names: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle with a fixed seed and cut off the first `val_fraction` for validation."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    val_size = round(len(order) * val_fraction)
    return order[val_size:], order[:val_size]


def write_split(rows: list[dict], data_dir: Path, split: str) -> None:
    """Write each photo to images/<split>/ and its boxes to labels/<split>/, the layout YOLO expects."""
    image_dir, label_dir = data_dir / "images" / split, data_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = Path(row["image"]["path"]).stem
        (image_dir / f"{name}.jpg").write_bytes(row["image"]["bytes"])
        lines = label_lines(row["objects"], row["width"], row["height"])
        (label_dir / f"{name}.txt").write_text("\n".join(lines) + "\n" if lines else "")


def write_dataset_yaml(data_dir: Path) -> Path:
    path = data_dir / "hardhat.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    path.write_text(
        f"path: {data_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n"
    )
    return path


def prepare_dataset(data_dir: str | Path = "data", val_fraction: float = 0.1, seed: int = 42) -> Path:
    """Download and convert the dataset on first use, and return the path of its YOLO
    dataset file. The official test split is kept whole; validation comes out of train."""
    data_dir = Path(data_dir)
    yaml_path = data_dir / "hardhat.yaml"
    if yaml_path.exists():
        return yaml_path

    data_dir.mkdir(parents=True, exist_ok=True)
    for split, file in SOURCE_FILES.items():
        parquet = data_dir / file
        if not parquet.exists():
            print(f"downloading Hard Hat Workers {split} split...")
            urllib.request.urlretrieve(SOURCE_URL.format(file=file), parquet)
        rows = pq.read_table(parquet).to_pylist()
        if split == "train":
            by_name = {Path(row["image"]["path"]).stem: row for row in rows}
            train_names, val_names = split_names(list(by_name), val_fraction, seed)
            write_split([by_name[n] for n in train_names], data_dir, "train")
            write_split([by_name[n] for n in val_names], data_dir, "val")
        else:
            write_split(rows, data_dir, "test")
        parquet.unlink()

    return write_dataset_yaml(data_dir)


if __name__ == "__main__":
    print(prepare_dataset())
