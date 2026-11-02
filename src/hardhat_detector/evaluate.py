"""Evaluate trained detectors on both test sets, compare them, and draw the README figures.

    python -m hardhat_detector.evaluate
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from hardhat_detector.data import CLASSES, prepare_dataset
from hardhat_detector.draw import draw_detections
from hardhat_detector.metrics import SIZE_BUCKETS, alarm_metrics, match_ground_truth, recall_by_size
from hardhat_detector.predict import detect

MODEL_LABELS = {
    "yolov8n_hhw_only": "YOLOv8n, Hard Hat Workers only",
    "yolov8s_hhw_only": "YOLOv8s, Hard Hat Workers only",
    "yolov8n": "YOLOv8n, both datasets",
    "yolov8s": "YOLOv8s, both datasets",
}
TEST_SPLITS = {"test": "Hard Hat Workers test", "test_roboflow": "Roboflow test"}
HEAD = CLASSES.index("head")
COLOURS = {"yolov8n_hhw_only": "#a0aec0", "yolov8s_hhw_only": "#718096", "yolov8n": "#2b6cb0", "yolov8s": "#dd6b20"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_LABELS))
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for counting a detection")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--assets-dir", default="assets")
    return parser.parse_args()


def label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def read_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """A YOLO label file back into pixel (x1, y1, x2, y2) boxes and class ids."""
    rows = np.loadtxt(label_path, ndmin=2) if label_path.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), rows[:, 0].astype(int)


def load_split(data_dir: Path, split: str) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    samples = []
    for image_path in sorted((data_dir / "images" / split).glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size
        boxes, classes = read_labels(data_dir / "labels" / split / f"{image_path.stem}.txt", width, height)
        samples.append((image_path, boxes, classes))
    return samples


def latency_ms(model: YOLO, paths: list[Path], device: str, count: int) -> float:
    """Median time for the network itself on one photo at a time (no batching), after a warm-up."""
    for path in paths[:5]:
        model.predict(path, device=device, verbose=False)
    times = [model.predict(path, device=device, verbose=False)[0].speed["inference"] for path in paths[:count]]
    return float(np.median(times))


def evaluate_split(model: YOLO, data_yaml: Path, split: str, samples: list, conf: float) -> dict:
    # Standard detection metrics, computed by Ultralytics over every confidence level.
    box = model.val(data=str(data_yaml), split=split, batch=16, plots=False, verbose=False).box
    metrics = {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class": {c: {"AP50": float(box.ap50[i]), "AP50-95": float(box.ap[i])} for i, c in enumerate(CLASSES)},
    }

    # Our own metrics, at a single working confidence threshold, using the same detect() the demo uses.
    found, truth, raised = [], [], []
    for image_path, gt_boxes, gt_classes in samples:
        result = detect(model, image_path, conf)
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)
        found.append(match_ground_truth(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                        pred_classes, gt_boxes, gt_classes))
        truth.append(bool((gt_classes == HEAD).any()))
        raised.append(bool((pred_classes == HEAD).any()))
    metrics["recall_by_size"] = recall_by_size([s[1] for s in samples], found)
    # The failure that prompted the second dataset: bare heads that fill much of the photo.
    metrics["bare_head_recall_by_size"] = recall_by_size(
        [s[1][s[2] == HEAD] for s in samples], [f[s[2] == HEAD] for f, s in zip(found, samples)]
    )
    metrics["alarm"] = alarm_metrics(truth, raised)
    return metrics


def evaluate_model(name: str, data_yaml: Path, splits: dict[str, list], conf: float) -> dict:
    model = YOLO(f"checkpoints/{name}.pt")
    metrics = {"model": name, "parameters": sum(p.numel() for p in model.model.parameters()), "conf_threshold": conf}
    metrics["splits"] = {split: evaluate_split(model, data_yaml, split, samples, conf) for split, samples in splits.items()}
    paths = [s[0] for s in splits["test"]]
    metrics["gpu_ms"] = latency_ms(model, paths, "0", count=300)
    metrics["cpu_ms"] = latency_ms(model, paths, "cpu", count=100)
    return metrics


def grouped_bars(ax, results: list[dict], groups: list[str], value, group_labels: list[str]) -> None:
    width = 0.8 / len(results)
    for k, r in enumerate(results):
        x = np.arange(len(groups)) + (k - (len(results) - 1) / 2) * width
        values = [value(r, g) for g in groups]
        bars = ax.bar(x, values, width, label=label(r["model"]), color=COLOURS.get(r["model"]))
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_xticks(range(len(groups)), group_labels)
    ax.spines[["top", "right"]].set_visible(False)


def plot_before_after(results: list[dict], path: Path) -> None:
    """mAP50 overall and for bare heads, on both test sets."""
    splits = list(TEST_SPLITS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, value) in zip(axes, (
        ("All heads: mAP50", lambda r, s: 100 * r["splits"][s]["mAP50"]),
        ("Bare heads only: AP50", lambda r, s: 100 * r["splits"][s]["per_class"]["head"]["AP50"]),
    )):
        grouped_bars(ax, results, splits, value, [TEST_SPLITS[s] for s in splits])
        ax.set(title=title, ylim=(0, 110))
    axes[0].set_ylabel("%")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bare_head_recall(results: list[dict], split: str, path: Path) -> None:
    buckets = [name for name, _, _ in SIZE_BUCKETS]
    counts = results[0]["splits"][split]["bare_head_recall_by_size"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    grouped_bars(ax, results, buckets, lambda r, b: 100 * (r["splits"][split]["bare_head_recall_by_size"][b]["recall"] or 0),
                 [f"{b}\n({counts[b]['boxes']:,} bare heads)" for b in buckets])
    ax.set(ylabel="bare heads found (%)", ylim=(0, 110), title=f"Bare heads found, by size ({TEST_SPLITS[split]}, IoU 0.5)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(models: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in models:
        with (Path("results") / name / "results.csv").open() as f:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
        ax.plot([int(r["epoch"]) for r in rows], [100 * float(r["metrics/mAP50-95(B)"]) for r in rows],
                label=label(name), color=COLOURS.get(name))
    ax.set(xlabel="epoch", ylabel="validation mAP50-95 (%)", title="Validation mAP during training (merged validation set)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def is_close_up(sample) -> bool:
    """Whether any labelled head fills more than a tenth of the photo."""
    image_path, boxes, _ = sample
    with Image.open(image_path) as image:
        area = image.width * image.height
    return bool(len(boxes)) and bool((((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) > 0.1 * area).any())


def plot_samples(model: YOLO, samples: list, conf: float, path: Path, count: int = 6, seed: int = 3) -> None:
    """Random test photos with the model's detections: half close-ups, half wider shots."""
    rng = random.Random(seed)
    close = [s for s in samples if is_close_up(s)]
    wide = [s for s in samples if not is_close_up(s)]
    chosen = rng.sample(close, count // 2) + rng.sample(wide, count - count // 2)
    fig, axes = plt.subplots(2, count // 2, figsize=(count * 2.4, 6.4))
    for ax, (image_path, _, _) in zip(axes.flat, chosen):
        ax.imshow(draw_detections(Image.open(image_path).convert("RGB"), detect(model, image_path, conf)))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_yaml = prepare_dataset(data_dir)
    splits = {split: load_split(data_dir, split) for split in TEST_SPLITS}
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    models = [m for m in args.models if Path(f"checkpoints/{m}.pt").exists()]
    results = []
    for name in models:
        start = time.time()
        metrics = evaluate_model(name, data_yaml, splits, args.conf)
        (Path("results") / name).mkdir(parents=True, exist_ok=True)
        (Path("results") / name / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results.append(metrics)
        for split, m in metrics["splits"].items():
            large = m["bare_head_recall_by_size"]["large"]["recall"]
            print(f"{label(name):<32} {TEST_SPLITS[split]:<22} mAP50 {m['mAP50']:.3f} | mAP50-95 {m['mAP50-95']:.3f} | "
                  f"bare-head AP50 {m['per_class']['head']['AP50']:.3f} | large bare heads found {large:.3f} | "
                  f"alarm P {m['alarm']['precision']:.3f} R {m['alarm']['recall']:.3f}")
        print(f"{'':<32} GPU {metrics['gpu_ms']:.1f} ms | CPU {metrics['cpu_ms']:.0f} ms ({time.time() - start:.0f}s)")

    merged = [m for m in models if not m.endswith("_hhw_only")]
    plot_before_after(results, assets / "before_after.png")
    plot_bare_head_recall(results, "test_roboflow", assets / "bare_head_recall.png")
    plot_curves(merged, assets / "training_curves.png")
    best = max((r for r in results if r["model"] in merged), key=lambda r: r["splits"]["test_roboflow"]["mAP50-95"])
    plot_samples(YOLO(f"checkpoints/{best['model']}.pt"), splits["test_roboflow"], args.conf, assets / "sample_predictions.png")
    print(f"figures written to {assets}/")


if __name__ == "__main__":
    main()
