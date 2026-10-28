"""Run a recorded E00 smoke test or baseline on the frozen local dataset.

Python 3.9 compatible. No remote dataset/weight downloads are requested.
The smoke test uses random initialization; full runs require local weights.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys

from freeze_dataset import sha256
from materialize_dataset import read_checksums

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "8.3.235"


class PlannedPause(Exception):
    """End the checkpoint probe before upstream strips optimizer state."""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command(args):
    return subprocess.check_output(args, cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()


def select_rows(rows, split, count=None):
    if split not in ("train", "val"):
        raise ValueError("This training entry only reads train/val, never test")
    selected = sorted((r for r in rows if r["split"] == split), key=lambda r: r["stem"])
    if count is not None:
        if len(selected) < count:
            raise ValueError("Too few images in split: " + split)
        selected = random.Random(20261028).sample(selected, count)
    return selected


def make_data_yaml(data, frozen, run_dir, smoke):
    with (frozen / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads((frozen / "metadata.json").read_text(encoding="utf-8"))
    counts = {}
    lines = ["path: " + json.dumps((data / "detect").as_posix())]
    for split, count in (("train", 64), ("val", 32)):
        selected = select_rows(rows, split, count if smoke else None)
        paths = [data / "detect" / "images" / split / (r["stem"] + ".png") for r in selected]
        if not all(p.is_file() for p in paths):
            raise ValueError("Missing derived images in " + split)
        listing = run_dir / (split + "-images.txt")
        listing.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
        lines.append(split + ": " + json.dumps(listing.as_posix()))
        counts[split] = {"images": len(selected), "objects": sum(int(r["objects"]) for r in selected)}
    lines += ["nc: 5", "names: " + json.dumps(metadata["names"])]
    target = run_dir / "data.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, counts


def check_dataset_evidence(data, frozen, report):
    read_checksums(frozen)
    marker = json.loads((data / "BUILD_COMPLETE.json").read_text(encoding="utf-8"))
    validation = json.loads(report.read_text(encoding="utf-8"))
    digest = sha256(frozen / "manifest.csv")
    if marker["manifest_sha256"] != digest or validation["manifest_sha256"] != digest:
        raise ValueError("Dataset or validation report uses a different frozen manifest")
    if validation.get("status") != "passed" or validation.get("validated_image_pairs") != 14660:
        raise ValueError("Full paired dataset validation must pass first")
    return digest


def check_full_protocol(config, weights, seed, manifest_sha):
    lock = json.loads((REPO / "configs/e00_protocol_v2.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != lock["training_config_canonical_sha256"]:
        raise ValueError("Training config differs from the frozen E00 protocol; use a new protocol for changes")
    if manifest_sha != lock["dataset_manifest_sha256"] or seed not in lock["seeds"]:
        raise ValueError("Data or seed differs from the frozen E00 protocol")
    expected = lock["initialization"]
    if weights is None or weights.stat().st_size != expected["bytes"] or sha256(weights) != expected["sha256"]:
        raise ValueError("Initialization weights differ from the frozen official-release copy")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "resume-smoke", "full"), default="smoke")
    parser.add_argument("--data-root", type=Path, default=Path("/root/datasets/faircsar_local_v1"))
    parser.add_argument("--validation-report", type=Path, default=REPO / "outputs/server-data-validation.json")
    parser.add_argument("--name", required=True, help="Unique run directory name; existing directories are refused")
    parser.add_argument("--weights", type=Path, help="Local, verified YOLOv8s initialization; required for full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-stage", choices=("interrupt", "resume"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("Use a simple unique run name without directory separators")
    if args.probe_stage and args.mode != "smoke":
        parser.error("Checkpoint probe stages only work in smoke mode")
    if args.mode == "resume-smoke":
        if args.weights:
            parser.error("The checkpoint probe uses random initialization")
        child = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "smoke",
                 "--name", args.name, "--data-root", str(args.data_root),
                 "--validation-report", str(args.validation_report), "--seed", str(args.seed)]
        for stage in ("interrupt", "resume"):
            subprocess.run(child + ["--probe-stage", stage], cwd=REPO, check=True)
        return
    if args.mode == "full" and args.weights is None:
        parser.error("Full training requires --weights with a verified local initialization checkpoint")
    if args.weights is not None and not args.weights.is_file():
        parser.error("Local weights do not exist")
    if command(["git", "status", "--porcelain"]):
        raise ValueError("Commit tracked/untracked source changes before starting an experiment")
    frozen = REPO / ("splits/faircsar_local_v2" if args.mode == "full" else "splits/faircsar_local_v1")
    data = args.data_root.resolve()
    manifest_sha = check_dataset_evidence(data, frozen, args.validation_report)
    config = json.loads((REPO / "configs/e00_yolov8s.json").read_text(encoding="utf-8"))
    protocol = check_full_protocol(config, args.weights, args.seed, manifest_sha) if args.mode == "full" else None

    # Set before CUDA context creation, for deterministic CUDA matrix operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    if ultralytics.__version__ != UPSTREAM:
        raise ValueError("Expected ultralytics==" + UPSTREAM + ", got " + ultralytics.__version__)
    if torch.__version__ != "2.0.1+cu117" or torchvision.__version__ != "0.15.2+cu117":
        raise ValueError("Environment differs from the verified torch/torchvision CUDA 11.7 stack")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    run_dir = REPO / "runs" / args.name
    previous = None
    if args.probe_stage == "resume":
        previous = json.loads((run_dir / "run-record.json").read_text(encoding="utf-8"))
        if previous.get("status") != "paused_for_resume_probe":
            raise ValueError("Only the deliberately paused checkpoint probe can reuse a run directory")
        if previous["git_commit"] != command(["git", "rev-parse", "HEAD"]) or previous["manifest_sha256"] != manifest_sha:
            raise ValueError("Code or data changed between checkpoint probe stages")
        if sha256(run_dir / "weights/last.pt") != previous["resume_checkpoint_sha256"]:
            raise ValueError("Paused checkpoint changed before resume")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "preparing", "mode": args.mode, "experiment": "E00",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__, "ultralytics_path": ultralytics.__file__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "manifest_sha256": manifest_sha, "validation_report_sha256": sha256(args.validation_report),
        "config_sha256": sha256(REPO / "configs/e00_yolov8s.json"),
        "initialization": str(args.weights.resolve()) if args.weights else "random YOLOv8s (smoke only)",
        "initial_weights_sha256": sha256(args.weights) if args.weights else None,
        "protocol": protocol,
        "limits": "Single-seed baseline; not evidence of a point-guidance benefit." if protocol else
                  "A smoke result verifies the pipeline only; no detection performance conclusion.",
    }
    if previous:
        record = previous
        record.update(status="resuming", resume_command=sys.argv)
    write_json(run_dir / "run-record.json", record)
    try:
        (run_dir / "pip-freeze.txt").write_text(command([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8")
        (run_dir / "nvidia-smi.txt").write_text(command(["nvidia-smi"]) + "\n", encoding="utf-8")
        package = Path(ultralytics.__file__).parent
        # Snapshot exact installed code, not just the package version string.
        code_hashes = {p.relative_to(package).as_posix(): sha256(p) for p in sorted(package.rglob("*"))
                       if p.is_file() and p.suffix in (".py", ".yaml")}
        write_json(run_dir / "upstream-source-sha256.json", code_hashes)
        data_yaml, expected = make_data_yaml(data, frozen, run_dir, args.mode == "smoke")
        config.update(data=str(data_yaml), project=str(run_dir.parent), name=args.name,
                      exist_ok=True, seed=args.seed, pretrained=bool(args.weights))
        if args.mode == "smoke":
            config.update(epochs=3 if args.probe_stage else 2, warmup_epochs=0.0)
        write_json(run_dir / "requested-train-args.json", config)
        record["expected_loader_counts"] = expected
        init_seeds(args.seed, deterministic=True)
        # Build architecture from packaged YAML, then transfer only matching weights.
        model = YOLO("yolov8s.yaml", task="detect")
        if args.weights:
            model.load(str(args.weights.resolve()))

        def audit_loaders(trainer):
            actual = {}
            for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
                labels = loader.dataset.labels
                actual[split] = {"images": len(loader.dataset), "objects": sum(len(r["cls"]) for r in labels)}
            write_json(run_dir / "loader-counts.json", actual)
            if actual != expected:
                raise RuntimeError("Framework dropped images/objects: " + repr(actual) + " != " + repr(expected))
            if args.probe_stage == "resume":
                if trainer.start_epoch != 1 or not trainer.optimizer.state or trainer.ema.updates <= 0:
                    raise RuntimeError("Resume failed to restore epoch, optimizer state, or EMA updates")
                record.update(resumed_from_epoch=trainer.start_epoch, resumed_optimizer_states=len(trainer.optimizer.state),
                              resumed_ema_updates=trainer.ema.updates)
            (run_dir / "train-transforms.txt").write_text(str(trainer.train_loader.dataset.transforms), encoding="utf-8")
            record["status"] = "running"
            write_json(run_dir / "run-record.json", record)

        if args.probe_stage == "resume":
            model = YOLO(str(run_dir / "weights/last.pt"))
        model.add_callback("on_train_start", audit_loaders)

        def pause_after_checkpoint(trainer):
            if trainer.epoch == 0:
                # A completed training checkpoint has its optimizer stripped. Interrupt
                # after the first normal checkpoint instead of resuming a finished run.
                checkpoint = torch.load(trainer.last, map_location="cpu")
                if checkpoint["epoch"] != 0 or not checkpoint.get("optimizer", {}).get("state"):
                    raise RuntimeError("First checkpoint lacks resumable optimizer state")
                record.update(status="paused_for_resume_probe", resume_checkpoint_sha256=sha256(trainer.last))
                write_json(run_dir / "run-record.json", record)
                raise PlannedPause()

        if args.probe_stage == "interrupt":
            model.add_callback("on_model_save", pause_after_checkpoint)
        try:
            if args.probe_stage == "resume":
                model.train(resume=True)
            else:
                model.train(**config)
        except PlannedPause:
            print("Checkpoint saved after epoch 1; starting a fresh process to resume.", flush=True)
            return
        weights = {}
        for name in ("best.pt", "last.pt"):
            checkpoint = run_dir / "weights" / name
            if not checkpoint.is_file():
                raise RuntimeError("Missing checkpoint: " + str(checkpoint))
            weights[name] = sha256(checkpoint)
        with (run_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
        if len(results) != config["epochs"]:
            raise RuntimeError("Training did not finish all requested epochs")
        for row in results:
            for key, value in row.items():
                if "loss" in key and not math.isfinite(float(value)):
                    raise RuntimeError("Non-finite loss in results.csv")
        # Reload saved weights and run one validation image through inference.
        image = Path((run_dir / "val-images.txt").read_text(encoding="utf-8").splitlines()[0])
        prediction = YOLO(str(run_dir / "weights/best.pt")).predict(
            source=str(image), imgsz=config["imgsz"], device="0", save=False, verbose=False)
        if len(prediction) != 1:
            raise RuntimeError("Saved checkpoint inference did not return one image result")
        record.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      checkpoint_sha256=weights, saved_checkpoint_inference="passed")
        write_json(run_dir / "run-record.json", record)
        print("PASS:", run_dir, flush=True)
    except BaseException as error:
        record.update(status="failed", error=type(error).__name__ + ": " + str(error))
        write_json(run_dir / "run-record.json", record)
        raise


if __name__ == "__main__":
    main()
