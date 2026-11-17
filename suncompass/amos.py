"""Import the AMOS pilot manifest and export portable, byte-preserving datasets.

Source columns follow camera257_pilot/dataset/manifest.csv. The two regression
targets are copied as forward/left projections, never normalized to unit length.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

import numpy as np
from PIL import Image
from .preprocessing import contract

FRAME = "yaw_aligned_level_FLU"
SCHEMA = "suncompass-amos-v1"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def source_path(root, relative):
    relative = PurePosixPath(str(relative))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or not relative.parts:
        raise ValueError(f"Expected a safe relative image path: {relative}")
    path = (Path(root) / str(relative)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Image escapes image root: {relative}")
    if path.suffix.lower() not in EXTENSIONS or not path.is_file():
        raise ValueError(f"Missing or unsupported image: {path}")
    return path


def capture_time(value):
    """Parse a timezone-explicit source timestamp; this does not certify capture time."""
    value = str(value).strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.strptime(re.sub(r"\s*\([^)]*\)$", "", value), "%a %b %d %Y %H:%M:%S GMT%z")
    if parsed.tzinfo is None:
        raise ValueError("Capture timestamp needs an explicit timezone")
    return parsed.astimezone(timezone.utc)


def row_timestamp(row):
    return row.get("capture_timestamp_utc") or row.get("timestamp_utc")


def media_details(root, row):
    """Prefer native originals; independently check supplied preletterboxed PNGs.

    The rebuilt package's black-padded PNGs are reference artifacts. Training
    copies native JPEGs so the same shared letterbox applies to raw inference.
    """
    relative = row.get("original_relative_path") or row["relative_image_path"]
    path = source_path(root, relative)
    digest = sha256(path)
    expected = row.get("original_jpeg_sha256") if row.get("original_relative_path") else row.get("processed_jpeg_sha256") or row.get("image_sha256")
    if expected and digest != expected.lower():
        raise ValueError("Source image checksum mismatch")
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    keys = ("native_width", "native_height") if row.get("original_relative_path") else ("processed_width", "processed_height")
    for key, actual in zip(keys, (width, height)):
        if row.get(key) not in (None, "") and int(row[key]) != actual:
            raise ValueError(f"{key} disagrees with image")
    result = dict(relative_image_path=relative, sha256=digest, width=width, height=height)
    if row.get("original_relative_path"):
        required = ("original_jpeg_sha256", "letterboxed_png_sha256", "input_width", "input_height",
                    "letterbox_scale", "padding_left", "padding_top", "padding_right", "padding_bottom", "crop_applied")
        if any(row.get(key) in (None, "") for key in required):
            raise ValueError("Missing native/preletterbox provenance")
        if row["crop_applied"].lower() not in ("false", "0") or float(row["letterbox_scale"]) != 1:
            raise ValueError("Expected a no-crop, lossless scale-one reference")
        prepared = source_path(root, row["relative_image_path"])
        prepared_sha = sha256(prepared)
        if prepared_sha != row["letterboxed_png_sha256"].lower():
            raise ValueError("Prepared PNG checksum mismatch")
        left, top, right, bottom = (int(row["padding_" + key]) for key in ("left", "top", "right", "bottom"))
        if min(left, top, right, bottom) < 0 or (left+width+right, top+height+bottom) != (1024,1024):
            raise ValueError("Padding/native dimensions do not fill the 1024 canvas")
        if (left,top) != ((1024-width)//2,(1024-height)//2):
            raise ValueError("Reference letterbox is not centered")
        if (int(row["input_width"]),int(row["input_height"])) != (1024,1024):
            raise ValueError("Expected 1024-square reference inputs")
        with Image.open(path) as original, Image.open(prepared) as reference:
            if reference.size != (1024,1024) or reference.mode != "RGB":
                raise ValueError("Expected 1024-square RGB reference PNG")
            pixels = np.array(reference)
            expected_pixels = np.zeros((1024,1024,3),dtype=np.uint8)
            expected_pixels[top:top+height,left:left+width] = np.array(original.convert("RGB"))
            if not np.array_equal(pixels,expected_pixels):
                raise ValueError("Reference pixels/padding differ from the full native image")
        result["prepared_image"] = dict(relative_image_path=row["relative_image_path"], sha256=prepared_sha,
                                        size=[1024,1024], padding_rgb=[0,0,0], pixel_content_verified=True)
    return result


def validate_labels(row):
    if row.get("target_frame") != FRAME:
        raise ValueError(f"Expected target_frame={FRAME}; do not guess a camera frame")
    f, left = float(row["sun_f"]), float(row["sun_l"])
    norm = math.hypot(f, left)
    if not all(map(math.isfinite, (f, left))) or not 1e-8 < norm <= 1.00001:
        raise ValueError("Invalid forward/left unit-vector projection")
    if row.get("sun_u") not in (None, ""):
        up = float(row["sun_u"])
        if not math.isfinite(up) or abs(f*f + left*left + up*up - 1) > 1e-5:
            raise ValueError("Full FLU vector must have unit norm")
    fields = ("solar_azimuth_deg", "solar_elevation_deg", "heading_deg_clockwise_from_north")
    if all(row.get(key) not in (None, "") for key in fields):
        azimuth, elevation, heading = (math.radians(float(row[key])) for key in fields)
        expected = (math.cos(elevation)*math.cos(azimuth-heading),
                    -math.cos(elevation)*math.sin(azimuth-heading))
        if not all(math.isfinite(v) for v in expected) or max(abs(a-b) for a, b in zip((f, left), expected)) > 1e-5:
            raise ValueError("Labels disagree with declared solar/heading convention")
    if not str(row.get("label_status", "")).strip():
        raise ValueError("Missing label_status provenance")
    return f, left


def normalize_manifest(manifest, image_root=None):
    manifest = Path(manifest).resolve()
    if manifest.suffix.lower() == ".json":
        data = json.loads(manifest.read_text())
        if data.get("schema") != SCHEMA or data.get("target_frame") != FRAME or data.get("target_order") != ["sun_f", "sun_l"]:
            raise ValueError("Expected an imported SunCompass AMOS JSON manifest")
        if image_root is not None:
            root = Path(image_root).resolve()
        else:
            root = Path(data["image_root"])
            root = root.resolve() if root.is_absolute() else (manifest.parent / root).resolve()
        # Recheck the actual media, not just a prior import receipt.
        for item in data["images"]:
            path = source_path(root, item["relative_image_path"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Changed source image: {path}")
            row = item["metadata"]
            if (item["sun_f"], item["sun_l"]) != validate_labels(row):
                raise ValueError("Imported labels disagree with their source metadata")
            if item["split"] != row["split"].strip() or item["camera_id"] != row["camera_id"].strip() or capture_time(row_timestamp(item)) != capture_time(row_timestamp(row)):
                raise ValueError("Imported identity/split disagrees with source metadata")
            with Image.open(path) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError("Imported dimensions disagree with image")
                image.verify()
            if row.get("original_relative_path") and "source_relative_image_path" not in item:
                details = media_details(root,row)
                if any(item.get(key) != value for key,value in details.items()):
                    raise ValueError("Imported native/reference provenance is inconsistent")
        data["image_root"] = str(root)
        validate_splits(data["images"])
        return data
    if manifest.suffix.lower() != ".csv":
        raise ValueError("Use the package's dataset/manifest.csv, or an imported JSON manifest")
    root = Path(image_root or manifest.parent).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_image_path", "camera_id", "split",
                    "sun_f", "sun_l", "target_frame", "label_status"}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest needs columns {sorted(required)}")
        if not {"capture_timestamp_utc", "timestamp_utc"}.intersection(reader.fieldnames):
            raise ValueError("Manifest needs a timezone-explicit capture or archive timestamp")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty AMOS manifest")
    images = []
    paths = set()
    for number, row in enumerate(rows, 2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row")
            details = media_details(root,row)
            relative = details["relative_image_path"]
            path = source_path(root, relative)
            if str(path) in paths:
                raise ValueError("Duplicate source image")
            paths.add(str(path))
            f, left = validate_labels(row)
            camera = row["camera_id"].strip()
            if not camera:
                raise ValueError("Missing camera_id")
            timestamp = capture_time(row_timestamp(row)).isoformat()
            images.append(dict(id=hashlib.sha256(relative.encode()).hexdigest()[:24],
                camera_id=camera, timestamp_utc=timestamp,
                timestamp_role="capture_estimate" if row.get("capture_timestamp_utc") else "archive",
                split=row["split"].strip(), sun_f=f, sun_l=left, metadata=row, **details))
        except (ValueError, KeyError, OSError) as error:
            raise ValueError(f"{manifest}:{number}: {error}") from error
    images.sort(key=lambda item: item["relative_image_path"])
    validate_splits(images)
    return dict(schema=SCHEMA, source_manifest_sha256=sha256(manifest),
                image_root=str(root), target_frame=FRAME, target_order=["sun_f", "sun_l"],
                split_policy="preserve supplied camera/source-date holdouts", images=images)


def validate_splits(images):
    if not images or len({item["id"] for item in images}) != len(images):
        raise ValueError("Empty manifest or duplicate image IDs")
    groups, hashes = {}, {}
    for item in images:
        if not re.fullmatch(r"[a-f0-9]{24}", str(item["id"])):
            raise ValueError("Invalid imported image ID")
        split = item["split"]
        if split not in ("train", "test", "val"):
            raise ValueError(f"Unknown split: {split}")
        day = (item["camera_id"], capture_time(row_timestamp(item)).date().isoformat())
        if groups.setdefault(day, split) != split:
            raise ValueError(f"Camera/capture date leaks across splits: {day}")
        if hashes.setdefault(item["sha256"], split) != split:
            raise ValueError("Identical image bytes leak across splits")
    counts = Counter(item["split"] for item in images)
    if not counts["train"] or not (counts["test"] or counts["val"]):
        raise ValueError("Manifest needs training and held-out images")


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["image_path", "sun_f", "sun_l"])
    writer.writerows(rows)
    return stream.getvalue().encode()


def verify_export(root):
    root = Path(root).resolve()
    receipt = json.loads((root / "export.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("preprocessing") != contract():
        raise ValueError("Unknown export/preprocessing contract")
    found = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Export must contain materialized files, not symlinks")
    if found != set(receipt["files"]) | {"export.json"}:
        raise ValueError("Missing or unexpected export files")
    for relative, expected in receipt["files"].items():
        path = source_path(root, relative) if relative.startswith(("train/", "test/", "val/")) else root / relative
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Changed export file: {relative}")
    return receipt


def export_dataset(manifest, output, image_root=None):
    data = normalize_manifest(manifest, image_root)
    fingerprint = hashlib.sha256(encoded({key: value for key, value in data.items() if key != "image_root"})).hexdigest()
    output = Path(output).absolute()
    if output.exists():
        receipt = verify_export(output)
        if receipt["import_fingerprint"] != fingerprint:
            raise FileExistsError("Output belongs to a different manifest; choose a new output directory")
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        exported = []
        for item in data["images"]:
            source = source_path(data["image_root"], item["relative_image_path"])
            relative = f"{item['split']}/images/{item['id']}{source.suffix.lower()}"
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(destination) != item["sha256"]:
                raise ValueError(f"Source changed during export: {source}")
            exported.append(dict(item, source_relative_image_path=item["relative_image_path"], relative_image_path=relative))
        for split in sorted({item["split"] for item in exported}):
            rows = [(item["relative_image_path"], format(item["sun_f"], ".17g"), format(item["sun_l"], ".17g"))
                    for item in exported if item["split"] == split]
            (stage / f"{split}.csv").write_bytes(csv_bytes(rows))
        (stage / "manifest.json").write_bytes(encoded(dict(data, image_root=".", images=exported)))
        files = {str(path.relative_to(stage)): sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        receipt = dict(schema=SCHEMA, import_fingerprint=fingerprint,
                       source_manifest_sha256=data["source_manifest_sha256"],
                       preprocessing=contract(), target_order=["sun_f", "sun_l"],
                       counts=dict(sorted(Counter(item["split"] for item in exported).items())), files=files)
        (stage / "export.json").write_bytes(encoded(receipt))
        verify_export(stage)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
        return receipt
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("import", "export"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--image-root", type=Path)
        command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("verify")
    command.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import":
        data = normalize_manifest(args.manifest, args.image_root)
        payload = encoded(data)
        if args.output.exists() and args.output.read_bytes() != payload:
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.exists():
            with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                os.link(temporary, args.output)  # publish atomically without overwriting
            finally:
                temporary.unlink()
        result = dict(schema=SCHEMA, images=len(data["images"]), manifest=str(args.output))
    else:
        receipt = verify_export(args.dataset) if args.command == "verify" else export_dataset(args.manifest, args.output, args.image_root)
        result = dict(schema=SCHEMA, verified=True, counts=receipt["counts"], import_fingerprint=receipt["import_fingerprint"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
