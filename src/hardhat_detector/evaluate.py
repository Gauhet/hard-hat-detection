"""Evaluate trained detectors on the test set, compare them, and draw the README figures.

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

MODEL_LABELS = {"yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s", "yolov8m": "YOLOv8m"}
HEAD = CLASSES.index("head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolov8m"])
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for counting a detection")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--assets-dir", default="assets")
    return parser.parse_args()


def read_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """A YOLO label file back into pixel (x1, y1, x2, y2) boxes and class ids."""
    rows = np.loadtxt(label_path, ndmin=2) if label_path.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), rows[:, 0].astype(int)


def test_set(data_dir: Path) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    samples = []
    for image_path in sorted((data_dir / "images" / "test").glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size
        boxes, classes = read_labels(data_dir / "labels" / "test" / f"{image_path.stem}.txt", width, height)
        samples.append((image_path, boxes, classes))
    return samples


def latency_ms(model: YOLO, paths: list[Path], device: str, count: int) -> float:
    """Median time for the network itself on one photo at a time (no batching), after a warm-up."""
    for path in paths[:5]:
        model.predict(path, device=device, verbose=False)
    times = [model.predict(path, device=device, verbose=False)[0].speed["inference"] for path in paths[:count]]
    return float(np.median(times))


def evaluate_model(name: str, data_yaml: Path, samples: list, conf: float) -> dict:
    model = YOLO(f"checkpoints/{name}.pt")

    # Standard detection metrics, computed by Ultralytics over every confidence level.
    box = model.val(data=str(data_yaml), split="test", batch=16, plots=False, verbose=False).box
    metrics = {
        "model": name,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class": {c: {"AP50": float(box.ap50[i]), "AP50-95": float(box.ap[i])} for i, c in enumerate(CLASSES)},
    }

    # Our own metrics, at a single working confidence threshold.
    found, truth, raised = [], [], []
    for image_path, gt_boxes, gt_classes in samples:
        result = model.predict(image_path, conf=conf, verbose=False)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)
        found.append(match_ground_truth(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                        pred_classes, gt_boxes, gt_classes))
        truth.append(bool((gt_classes == HEAD).any()))
        raised.append(bool((pred_classes == HEAD).any()))
    metrics["conf_threshold"] = conf
    metrics["recall_by_size"] = recall_by_size([s[1] for s in samples], found)
    metrics["alarm"] = alarm_metrics(truth, raised)

    paths = [s[0] for s in samples]
    metrics["gpu_ms"] = latency_ms(model, paths, "0", count=300)
    metrics["cpu_ms"] = latency_ms(model, paths, "cpu", count=100)
    return metrics


def plot_comparison(results: list[dict], path: Path) -> None:
    """Accuracy and speed side by side: the trade-off you pick a model size by."""
    names = [MODEL_LABELS.get(r["model"], r["model"]) for r in results]
    colours = ["#2b6cb0", "#dd6b20", "#38a169"][: len(results)]
    panels = (
        ("mAP50-95", lambda r: 100 * r["mAP50-95"], "test mAP50-95 (%)", "{:.1f}", (0, 100)),
        ("GPU", lambda r: r["gpu_ms"], "ms per photo, GPU", "{:.1f}", None),
        ("CPU", lambda r: r["cpu_ms"], "ms per photo, CPU", "{:.0f}", None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (title, value, ylabel, fmt, ylim) in zip(axes, panels):
        values = [value(r) for r in results]
        bars = ax.bar(names, values, color=colours)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3)
        ax.set(ylabel=ylabel, title=title)
        ax.set_ylim(*(ylim or (0, max(values) * 1.2)))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same accuracy, but the small model is much faster on a CPU")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(models: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in models:
        with (Path("results") / name / "results.csv").open() as f:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
        ax.plot([int(r["epoch"]) for r in rows], [100 * float(r["metrics/mAP50-95(B)"]) for r in rows],
                label=MODEL_LABELS.get(name, name))
    ax.set(xlabel="epoch", ylabel="validation mAP50-95 (%)", title="Validation mAP during training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_recall_by_size(results: list[dict], path: Path) -> None:
    buckets = [name for name, _, _ in SIZE_BUCKETS]
    counts = results[0]["recall_by_size"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.8 / len(results)
    for k, r in enumerate(results):
        x = np.arange(len(buckets)) + (k - (len(results) - 1) / 2) * width
        bars = ax.bar(x, [100 * r["recall_by_size"][b]["recall"] for b in buckets], width,
                      label=MODEL_LABELS.get(r["model"], r["model"]))
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_xticks(range(len(buckets)), [f"{b}\n({counts[b]['boxes']:,} heads)" for b in buckets])
    ax.set(ylabel="heads found (%)", ylim=(0, 105), title="Recall by head size (IoU 0.5)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(model: YOLO, samples: list, conf: float, path: Path, count: int = 6, seed: int = 8) -> None:
    chosen = random.Random(seed).sample(samples, count)
    fig, axes = plt.subplots(2, count // 2, figsize=(count * 2.4, 6.4))
    for ax, (image_path, _, _) in zip(axes.flat, chosen):
        result = model.predict(image_path, conf=conf, verbose=False)[0]
        ax.imshow(draw_detections(Image.open(image_path).convert("RGB"), result))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_yaml = prepare_dataset(args.data_dir)
    samples = test_set(Path(args.data_dir))
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    models = [m for m in args.models if Path(f"checkpoints/{m}.pt").exists()]
    results = []
    for name in models:
        start = time.time()
        metrics = evaluate_model(name, data_yaml, samples, args.conf)
        (Path("results") / name / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results.append(metrics)
        alarm = metrics["alarm"]
        print(f"{MODEL_LABELS.get(name, name):<8} mAP50 {metrics['mAP50']:.3f} | mAP50-95 {metrics['mAP50-95']:.3f} | "
              f"GPU {metrics['gpu_ms']:.1f} ms | CPU {metrics['cpu_ms']:.0f} ms | alarm P {alarm['precision']:.3f} "
              f"R {alarm['recall']:.3f} ({time.time() - start:.0f}s)")

    best = max(results, key=lambda r: r["mAP50-95"])
    plot_comparison(results, assets / "comparison.png")
    plot_curves(models, assets / "training_curves.png")
    plot_recall_by_size(results, assets / "recall_by_size.png")
    plot_samples(YOLO(f"checkpoints/{best['model']}.pt"), samples, args.conf, assets / "sample_predictions.png")
    print(f"figures written to {assets}/")


if __name__ == "__main__":
    main()
