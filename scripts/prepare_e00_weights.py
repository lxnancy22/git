"""Download a fixed official YOLOv8s release and record the local artifact identity.

The release API does not publish a digest for this older asset. The calculated
SHA-256 pins our downloaded copy; it is not an upstream-published checksum.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from freeze_dataset import sha256

REPO = Path(__file__).resolve().parents[1]
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt"
SIZE = 22588772  # Official release API asset size checked 2026-10-28.


def check_locked_copy(path):
    lock = json.loads((REPO / "configs/e00_protocol_v1.json").read_text(encoding="utf-8"))["initialization"]
    if path.stat().st_size != lock["bytes"] or sha256(path) != lock["sha256"]:
        raise ValueError("Artifact differs from the frozen E00 initialization copy")


def main():
    target = REPO / "weights/pretrained/v8.3.0/yolov8s.pt"
    metadata_path = target.with_suffix(".source.json")
    if target.exists() or metadata_path.exists():
        if not target.is_file() or not metadata_path.is_file():
            raise ValueError("Incomplete existing artifact; inspect before retrying")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["url"] != URL or metadata["sha256"] != sha256(target) or target.stat().st_size != SIZE:
            raise ValueError("Existing artifact differs from its source record")
        check_locked_copy(target)
        print("Already verified:", target)
        print("SHA256:", metadata["sha256"])
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pt.partial")
    # Exclusive creation prevents accidentally overwriting earlier failed downloads.
    try:
        with partial.open("xb") as output:
            request = urllib.request.Request(URL, headers={"User-Agent": "SAR-research-setup/1.0"})
            print("Downloading official release:", URL, flush=True)
            with urllib.request.urlopen(request, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
    except FileExistsError:
        raise ValueError("Partial download already exists; inspect it before retrying: " + str(partial))
    except BaseException:
        # Remove only this invocation's incomplete download; never touch old weights.
        partial.unlink(missing_ok=True)
        raise
    if partial.stat().st_size != SIZE:
        raise ValueError("Downloaded size differs from official release metadata; partial file retained")
    digest = sha256(partial)
    check_locked_copy(partial)
    partial.rename(target)
    metadata_path.write_text(json.dumps({
        "url": URL, "release": "v8.3.0", "name": "yolov8s.pt", "bytes": SIZE,
        "sha256": digest, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "role": "COCO pretrained initialization candidate; not a SAR-trained legacy checkpoint",
        "digest_basis": "Locally calculated after official HTTPS download; upstream API digest was null"
    }, indent=2) + "\n", encoding="utf-8")
    print("READY:", target)
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
