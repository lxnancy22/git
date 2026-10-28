"""Checks for split leakage and loader inputs without requiring GPU packages."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from train_e00 import REPO, check_dataset_evidence, make_data_yaml, select_rows
from freeze_dataset import sha256


class TrainingProtocolTests(unittest.TestCase):
    def test_test_and_quarantine_are_rejected(self):
        for split in ("test", "quarantine"):
            with self.assertRaises(ValueError):
                select_rows([], split)

    def test_smoke_sampling_preserves_partition_and_is_reproducible(self):
        rows = [{"split": "train", "stem": str(i)} for i in range(100)]
        rows += [{"split": "test", "stem": "forbidden"}]
        selected = select_rows(rows, "train", 64)
        self.assertEqual(selected, select_rows(list(reversed(rows)), "train", 64))
        self.assertEqual(len({r["stem"] for r in selected}), 64)
        self.assertTrue(all(r["split"] == "train" for r in selected))

    def test_smoke_lists_point_to_original_split_images(self):
        frozen = REPO / "splits/faircsar_local_v1"
        with (frozen / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            for split, count in (("train", 64), ("val", 32)):
                folder = root / "detect/images" / split
                folder.mkdir(parents=True)
                for row in select_rows(rows, split, count):
                    (folder / (row["stem"] + ".png")).touch()
            yaml_path, counts = make_data_yaml(root, frozen, run_dir, True)
            self.assertEqual(counts["train"]["images"], 64)
            self.assertEqual(counts["val"]["images"], 32)
            self.assertNotIn("test:", yaml_path.read_text(encoding="utf-8"))
            for split in ("train", "val"):
                listed = (run_dir / (split + "-images.txt")).read_text(encoding="utf-8").splitlines()
                self.assertTrue(all(Path(p).parent == root / "detect/images" / split for p in listed))

    def test_full_lists_have_frozen_train_and_val_counts(self):
        with (REPO / "splits/faircsar_local_v1/manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for split, images, objects in (("train", 9582, 55906), ("val", 2377, 13392)):
            selected = select_rows(rows, split)
            self.assertEqual(len(selected), images)
            self.assertEqual(sum(int(r["objects"]) for r in selected), objects)

    def test_failed_or_wrong_manifest_validation_is_rejected(self):
        frozen = REPO / "splits/faircsar_local_v1"
        digest = sha256(frozen / "manifest.csv")
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "BUILD_COMPLETE.json").write_text(json.dumps({"manifest_sha256": digest}), encoding="utf-8")
            report = data / "report.json"
            for status, manifest in (("failed", digest), ("passed", "wrong")):
                report.write_text(json.dumps({"status": status, "manifest_sha256": manifest,
                                              "validated_image_pairs": 14660}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    check_dataset_evidence(data, frozen, report)


if __name__ == "__main__":
    unittest.main()
