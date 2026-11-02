"""Fine-tune a COCO-pretrained YOLOv8 model to find heads with and without hard hats.

    python -m hardhat_detector.train --model yolov8n
    python -m hardhat_detector.train --model yolov8s --epochs 60
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from hardhat_detector.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8n", help="yolov8n, yolov8s, yolov8m, ...")
    parser.add_argument("--name", help="name for the run, checkpoint and results folder (default: the model name)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15, help="stop after this many epochs without a better val mAP")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="disk", choices=["disk", "ram", "none"],
                        help="keep decoded photos on disk or in memory between epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name or args.model
    data_yaml = prepare_dataset(args.data_dir)

    # Start from weights trained on COCO, whose 80 classes include "person", so the
    # network already knows what people look like and only has to learn the two head classes.
    model = YOLO(f"{args.model}.pt")
    start = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        cache=False if args.cache == "none" else args.cache,
        project="runs",
        name=name,
        exist_ok=True,
    )
    minutes = (time.time() - start) / 60

    # Keep the best weights and the per-epoch log; the rest of runs/ is scratch output.
    run_dir = Path(model.trainer.save_dir)
    Path("checkpoints").mkdir(exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", f"checkpoints/{name}.pt")
    results_dir = Path("results") / name
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "results.csv", results_dir / "results.csv")
    summary = {**vars(args), "epochs_run": model.trainer.epoch + 1, "training_minutes": round(minutes, 1)}
    (results_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved checkpoints/{name}.pt after {minutes:.1f} min")


if __name__ == "__main__":
    main()
