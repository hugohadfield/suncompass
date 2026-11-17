"""Data integrity, real pilot schema, split and loader contracts."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from suncompass.amos import export_dataset, normalize_manifest, sha256, verify_export
from suncompass.data_loading import RegressionTaskData, load_image_targets_from_csv


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    rows = []
    for split, day, color in (("train", 24, (250, 30, 50)), ("test", 25, (20, 240, 50))):
        path = root / split / "camera257" / "image,one.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (32, 22), color).save(path)
        rows.append(dict(image_path="/stale/location.jpg", relative_image_path=str(path.relative_to(root)),
            camera_id="257", capture_timestamp_utc=f"Sun Sep {day} 2006 05:10:00 GMT-0700 (Pacific Daylight Time)",
            split=split, sun_f="0.6", sun_l="-0.3", sun_u=str(np.sqrt(.55)),
            target_frame="yaw_aligned_level_FLU", label_status="coarse_pilot", processed_width="32",
            processed_height="22", processed_jpeg_sha256=sha256(path)))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return root, manifest, rows


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def test_export_portable_repeatable_and_exact_targets(package, tmp_path):
    root, manifest, _ = package
    output = tmp_path / "export"
    first = export_dataset(manifest, output)
    assert first == export_dataset(manifest, output)
    assert first == verify_export(output)
    second = export_dataset(manifest, tmp_path / "second")
    assert second == first
    moved = tmp_path / "relocated"
    shutil.copytree(output, moved)
    task = RegressionTaskData(image_folder_path=moved, resize_size=1024, preprocessing="letterbox", augment=False, batch_size=2, num_workers=0)
    inputs, targets = next(iter(task.trainloader))
    assert inputs.shape == (1, 3, 1024, 1024)
    np.testing.assert_array_equal(targets.numpy(), np.array([[.6, -.3]], dtype=np.float32))
    assert len(task.trainloader.dataset) == len(task.testloader.dataset) == 1
    data = json.loads((output / "manifest.json").read_text())
    for item in data["images"]:
        assert sha256(output / item["relative_image_path"]) == sha256(root / item["source_relative_image_path"])
        with (output / f"{item['split']}.csv").open(newline="") as stream:
            assert next(csv.reader(stream)) == ["image_path", "sun_f", "sun_l"]


@pytest.mark.parametrize("fault", ["checksum", "dimensions", "nan", "norm", "frame", "duplicate", "date_leak", "escape"])
def test_reject_invalid_source(package, tmp_path, fault):
    root, manifest, rows = package
    if fault == "checksum": rows[0]["processed_jpeg_sha256"] = "0" * 64
    elif fault == "dimensions": rows[0]["processed_width"] = "1024"
    elif fault == "nan": rows[0]["sun_f"] = "nan"
    elif fault == "norm": rows[0]["sun_f"] = "2"
    elif fault == "frame": rows[0]["target_frame"] = "unknown"
    elif fault == "duplicate": rows.append(dict(rows[0]))
    elif fault == "date_leak": rows[1]["capture_timestamp_utc"] = rows[0]["capture_timestamp_utc"]
    elif fault == "escape": rows[0]["relative_image_path"] = "../outside.png"
    write_rows(manifest, rows)
    with pytest.raises(ValueError): export_dataset(manifest, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_corrupt_existing_export_is_never_overwritten(package, tmp_path):
    _, manifest, _ = package
    output = tmp_path / "export"
    export_dataset(manifest, output)
    (output / "train.csv").write_text("corrupt")
    with pytest.raises(ValueError): export_dataset(manifest, output)
    assert (output / "train.csv").read_text() == "corrupt"


def test_imported_json_rechecks_source(package, tmp_path):
    _, manifest, _ = package
    data = normalize_manifest(manifest)
    imported = tmp_path / "imported.json"
    imported.write_text(json.dumps(data))
    output = tmp_path / "export"
    export_dataset(imported, output)
    file = Path(data["image_root"]) / data["images"][0]["relative_image_path"]
    file.write_bytes(b"changed")
    with pytest.raises(ValueError): export_dataset(imported, output)


@pytest.mark.parametrize("fault", ["order", "dimensions", "camera"])
def test_imported_metadata_is_consistent(package, tmp_path, fault):
    _, manifest, _ = package
    data = normalize_manifest(manifest)
    if fault == "order": data["target_order"] = ["sun_l", "sun_f"]
    elif fault == "dimensions": data["images"][0]["width"] = 99
    else: data["images"][0]["camera_id"] = "other"
    imported = tmp_path / "imported.json"; imported.write_text(json.dumps(data))
    with pytest.raises(ValueError): normalize_manifest(imported)


def test_explicit_relative_root_is_relative_to_working_directory(package, tmp_path, monkeypatch):
    root, manifest, _ = package
    imported = tmp_path / "imported.json"
    imported.write_text(json.dumps(normalize_manifest(manifest)))
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(); monkeypatch.chdir(elsewhere)
    assert normalize_manifest(imported, Path("../package"))["image_root"] == str(root)


@pytest.fixture
def native_package(package):
    root, manifest, rows = package
    for index,row in enumerate(rows):
        original = root / "originals" / f"native-{index}.jpg"
        original.parent.mkdir(exist_ok=True)
        Image.new("RGB",(1024,576),(30+index*80,75,140)).save(original)
        with Image.open(original) as image:
            boxed = Image.new("RGB",(1024,1024),(0,0,0)); boxed.paste(image.convert("RGB"),(0,224))
        prepared = root / row["relative_image_path"];boxed.save(prepared)
        row.pop("capture_timestamp_utc");row.pop("processed_jpeg_sha256")
        row.pop("processed_width");row.pop("processed_height")
        row.update(original_relative_path=str(original.relative_to(root)),original_jpeg_sha256=sha256(original),
                   letterboxed_png_sha256=sha256(prepared),timestamp_utc=f"2014-09-{24+index}T12:10:00+00:00",
                   timestamp_source="AMOS_archive_timestamp_no_capture_time_overlay_or_EXIF",
                   native_width="1024",native_height="576",input_width="1024",input_height="1024",
                   letterbox_scale="1",padding_left="0",padding_right="0",padding_top="224",padding_bottom="224",crop_applied="false")
    write_rows(manifest,rows)
    return root,manifest,rows


def test_native_originals_and_preboxed_references(native_package,tmp_path):
    root,manifest,rows=native_package
    data=normalize_manifest(manifest)
    assert all(item["timestamp_role"]=="archive" for item in data["images"])
    assert all((item["width"],item["height"])==(1024,576) for item in data["images"])
    assert all(item["prepared_image"]["pixel_content_verified"] for item in data["images"])
    output=tmp_path/"export";export_dataset(manifest,output)
    exported=json.loads((output/"manifest.json").read_text())
    for item in exported["images"]:
        assert sha256(output/item["relative_image_path"])==item["metadata"]["original_jpeg_sha256"]
        with Image.open(output/item["relative_image_path"]) as image:assert image.size==(1024,576)
    inputs,_=next(iter(RegressionTaskData(image_folder_path=output, resize_size=1024, preprocessing="letterbox", num_workers=0).testloader))
    assert inputs.shape==(1,3,1024,1024)


@pytest.mark.parametrize("fault",["original_hash","png_hash","padding","pixels","crop"])
def test_rebuilt_reference_errors_fail(native_package,fault):
    root,manifest,rows=native_package
    if fault=="original_hash":rows[0]["original_jpeg_sha256"]="0"*64
    elif fault=="png_hash":rows[0]["letterboxed_png_sha256"]="0"*64
    elif fault=="padding":rows[0]["padding_top"]="223"
    elif fault=="crop":rows[0]["crop_applied"]="true"
    else:
        path=root/rows[0]["relative_image_path"]
        with Image.open(path) as image:
            image.putpixel((0,0),(255,0,0));image.save(path)
        rows[0]["letterboxed_png_sha256"]=sha256(path)
    write_rows(manifest,rows)
    with pytest.raises(ValueError):normalize_manifest(manifest)


@pytest.mark.parametrize("row", ["a.png,1,0,0", "a.png,nan,0", "a.png,0,0", "a.png,1,0\na.png,1,0"])
def test_loader_rejects_ambiguous_targets(tmp_path, row):
    path = tmp_path / "train.csv"
    path.write_text("image_path,sun_f,sun_l\n" + row + "\n")
    with pytest.raises(ValueError): load_image_targets_from_csv(path)


def test_loader_refuses_silent_image_drops(package, tmp_path):
    _, manifest, _ = package
    output = tmp_path / "export"; export_dataset(manifest, output)
    Image.new("RGB", (20, 20)).save(output / "train/images/unlabelled.png")
    with pytest.raises(ValueError, match="no target"):
        RegressionTaskData(image_folder_path=output).trainloader


def test_training_consumes_normalized_letters_once(package, tmp_path, monkeypatch):
    import torch
    from suncompass import pretrained_resnet as training
    class Capture(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.head = torch.nn.Linear(3, 2)
            self.seen = []
        def forward(self, inputs, preprocessed=False):
            assert preprocessed and inputs.shape[1:] == (3,1024,1024)
            self.seen.append(tuple(inputs.shape))
            return self.head(inputs.mean(dim=(2,3)))
    _, manifest, _ = package
    output = tmp_path / "export"; export_dataset(manifest, output)
    monkeypatch.setattr(training, "ResNetRegression", Capture)
    model = training.train_network("cpu", image_folder_path=output, batch_size=1,
        num_workers=0, amp=False, max_steps=1, log_dir=str(tmp_path / "logs"))
    assert model.seen == [(1,3,1024,1024)]
