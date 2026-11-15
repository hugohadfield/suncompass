#!/usr/bin/env python3
"""Evaluate SunCompass rotational consistency on 360 equirectangular panoramas.

This can run on either:
  * a single standard 2:1 equirectangular panorama, or
  * downloaded Mapillary Metropolis CAM_EQUIRECTANGULAR images.

For each panorama, the script renders perspective views at regular yaw intervals,
runs the existing SunCompass checkpoint, and rotates each local prediction back
into panorama coordinates. A perfectly rotation-equivariant model should predict
the same panorama-frame sun azimuth from every virtual view.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from suncompass import SunCompass


def wrap_degrees(angle_deg):
    """Wrap angles to [-180, 180)."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def circular_mean_degrees(angles_deg):
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return float(
        np.rad2deg(
            np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        )
    )


def circular_resultant_length(angles_deg):
    """Return circular concentration R in [0, 1]; 1 means perfect agreement."""
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return float(
        np.hypot(np.sin(angles).mean(), np.cos(angles).mean())
    )


def _perspective_grid(yaw_deg, pitch_deg, hfov_deg, output_size, device):
    """Build a grid_sample grid for a rectilinear virtual camera.

    Conventions:
      * the middle of the panorama is yaw 0 degrees;
      * positive yaw turns left;
      * positive pitch looks up;
      * the virtual camera uses x=right, y=down, z=forward.
    """
    hfov_rad = math.radians(hfov_deg)
    focal = 0.5 * output_size / math.tan(0.5 * hfov_rad)

    coords = torch.arange(output_size, dtype=torch.float32, device=device)
    x = (coords - (output_size - 1) / 2.0) / focal
    y = (coords - (output_size - 1) / 2.0) / focal
    yy, xx = torch.meshgrid(y, x)

    rays = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    rays = F.normalize(rays, dim=-1)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    right = torch.tensor(
        [math.cos(yaw), 0.0, math.sin(yaw)],
        dtype=torch.float32,
        device=device,
    )
    forward_horizontal = torch.tensor(
        [-math.sin(yaw), 0.0, math.cos(yaw)],
        dtype=torch.float32,
        device=device,
    )
    forward = (
        math.cos(pitch) * forward_horizontal
        + torch.tensor(
            [0.0, -math.sin(pitch), 0.0],
            dtype=torch.float32,
            device=device,
        )
    )
    down = torch.cross(forward, right)
    forward = F.normalize(forward, dim=0)
    down = F.normalize(down, dim=0)

    pano_rays = (
        rays[..., 0:1] * right
        + rays[..., 1:2] * down
        + rays[..., 2:3] * forward
    )
    pano_rays = F.normalize(pano_rays, dim=-1)

    px, py, pz = pano_rays.unbind(dim=-1)
    longitude = torch.atan2(px, pz)
    latitude = torch.atan2(-py, torch.sqrt(px * px + pz * pz))

    grid_x = longitude / math.pi
    grid_y = -2.0 * latitude / math.pi
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


def perspective_view(
    panorama_rgb,
    yaw_deg,
    pitch_deg=0.0,
    hfov_deg=90.0,
    output_size=300,
):
    """Render a square perspective view from a 2:1 equirectangular panorama."""
    if panorama_rgb.ndim != 3 or panorama_rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB panorama")

    height, width, _ = panorama_rgb.shape
    if abs(float(width) / float(height) - 2.0) > 0.15:
        raise ValueError(
            "Expected a roughly 2:1 equirectangular image, got {}x{}".format(
                width, height
            )
        )
    if not 1.0 < hfov_deg < 179.0:
        raise ValueError("--hfov-deg must be between 1 and 179 degrees")

    pano = (
        torch.from_numpy(np.ascontiguousarray(panorama_rgb))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    # grid_sample does not wrap at the left/right panorama seam. Adding one
    # wrapped column at each edge gives bilinear interpolation the correct
    # neighbour across that seam.
    pano = torch.cat([pano[..., -1:], pano, pano[..., :1]], dim=-1)
    pano = pano.unsqueeze(0)
    padded_width = width + 2

    grid = _perspective_grid(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        hfov_deg=hfov_deg,
        output_size=output_size,
        device=pano.device,
    )

    # Re-map grid x from the original W pixels into the W+2 padded image.
    original_x = (grid[..., 0] + 1.0) * 0.5 * (width - 1)
    padded_x = original_x + 1.0
    grid[..., 0] = 2.0 * padded_x / (padded_width - 1) - 1.0

    view = F.grid_sample(
        pano,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return (
        view.squeeze(0)
        .permute(1, 2, 0)
        .mul(255.0)
        .clamp(0, 255)
        .byte()
        .numpy()
    )


def load_json(path):
    with path.open("r") as handle:
        return json.load(handle)


def find_metropolis_panoramas(dataroot, split, limit=None, stride=1):
    """Find Metropolis CAM_EQUIRECTANGULAR images without requiring the SDK."""
    split_root = dataroot / split
    required = [
        split_root / "sensor.json",
        split_root / "calibrated_sensor.json",
        split_root / "sample_data.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Metropolis metadata files: {}".format(", ".join(missing))
        )

    sensors = load_json(split_root / "sensor.json")
    calibrated = load_json(split_root / "calibrated_sensor.json")
    sample_data = load_json(split_root / "sample_data.json")

    equirect_sensor_tokens = set(
        row["token"]
        for row in sensors
        if row.get("modality") == "camera"
        and row.get("channel") == "CAM_EQUIRECTANGULAR"
    )
    calibrated_tokens = set(
        row["token"]
        for row in calibrated
        if row.get("sensor_token") in equirect_sensor_tokens
    )

    candidates = [
        row
        for row in sample_data
        if row.get("calibrated_sensor_token") in calibrated_tokens
    ]
    candidates.sort(key=lambda row: row.get("timestamp", 0))

    records = []
    for index, row in enumerate(candidates):
        if index % stride != 0:
            continue

        path = dataroot / row["filename"]
        if not path.exists():
            alternate = split_root / row["filename"]
            if alternate.exists():
                path = alternate
            else:
                continue

        records.append(
            {
                "path": path,
                "token": str(row.get("token", path.stem)),
                "timestamp": row.get("timestamp"),
            }
        )
        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise RuntimeError(
            "No CAM_EQUIRECTANGULAR images found. Check the dataset root, "
            "split, and that the image archive has been unpacked."
        )

    return records


def predict_batch(compass, views_rgb):
    """Run the existing SunCompass checkpoint on a batch of RGB views."""
    transformed = [compass.apply_transforms(view) for view in views_rgb]
    batch = torch.cat(transformed, dim=0).to(compass.device)
    with torch.no_grad():
        outputs = compass.model(batch)
    return outputs.detach().cpu().numpy()


def evaluate_panorama(
    compass,
    record,
    yaws_deg,
    pitch_deg,
    hfov_deg,
    view_size,
    save_views_dir=None,
):
    with Image.open(record["path"]) as image:
        panorama = np.asarray(image.convert("RGB"))

    views = [
        perspective_view(
            panorama,
            yaw_deg=yaw,
            pitch_deg=pitch_deg,
            hfov_deg=hfov_deg,
            output_size=view_size,
        )
        for yaw in yaws_deg
    ]
    predictions = predict_batch(compass, views)

    rows = []
    pano_azimuths = []

    pano_dir = None
    if save_views_dir is not None:
        pano_dir = save_views_dir / record["token"]
        pano_dir.mkdir(parents=True, exist_ok=True)

    for yaw, view, prediction in zip(yaws_deg, views, predictions):
        forward = float(prediction[0])
        left = float(prediction[1])
        local_azimuth = math.degrees(math.atan2(left, forward))

        # Both view yaw and SunCompass azimuth are positive-left.
        pano_azimuth = float(wrap_degrees(yaw + local_azimuth))
        magnitude = math.hypot(forward, left)
        pano_azimuths.append(pano_azimuth)

        row = {
            "panorama": str(record["path"]),
            "token": record["token"],
            "timestamp": (
                record["timestamp"] if record["timestamp"] is not None else ""
            ),
            "view_yaw_deg": float(yaw),
            "view_pitch_deg": float(pitch_deg),
            "hfov_deg": float(hfov_deg),
            "pred_forward": forward,
            "pred_left": left,
            "pred_magnitude": magnitude,
            "local_azimuth_deg": local_azimuth,
            "pano_azimuth_deg": pano_azimuth,
        }
        rows.append(row)

        if pano_dir is not None:
            yaw_string = "{:+07.1f}".format(yaw).replace("+", "p").replace("-", "m")
            Image.fromarray(view).save(
                pano_dir / ("yaw_{}.jpg".format(yaw_string)),
                quality=92,
            )

    consensus = circular_mean_degrees(pano_azimuths)
    abs_errors = np.abs(
        wrap_degrees(np.asarray(pano_azimuths, dtype=np.float64) - consensus)
    )

    for row, error in zip(rows, abs_errors):
        row["abs_consistency_error_deg"] = float(error)

    summary = {
        "panorama": str(record["path"]),
        "token": record["token"],
        "timestamp": record["timestamp"] if record["timestamp"] is not None else "",
        "num_views": len(rows),
        "consensus_pano_azimuth_deg": consensus,
        "mean_abs_consistency_error_deg": float(abs_errors.mean()),
        "median_abs_consistency_error_deg": float(np.median(abs_errors)),
        "p90_abs_consistency_error_deg": float(np.percentile(abs_errors, 90)),
        "resultant_length": circular_resultant_length(pano_azimuths),
        "mean_prediction_magnitude": float(
            np.mean([row["pred_magnitude"] for row in rows])
        ),
    }

    return rows, summary


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SunCompass around 360 equirectangular panoramas."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--panorama",
        type=Path,
        help="Path to one standard 2:1 equirectangular RGB image.",
    )
    source.add_argument(
        "--metropolis-root",
        type=Path,
        help="Root of a downloaded Mapillary Metropolis dataset.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum Metropolis panoramas to evaluate (default: 10).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth Metropolis panorama after timestamp sorting.",
    )
    parser.add_argument(
        "--yaw-step-deg",
        type=float,
        default=10.0,
        help="Spacing between virtual camera headings (default: 10 degrees).",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=0.0,
        help="Virtual camera pitch; positive looks up (default: 0).",
    )
    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=90.0,
        help="Horizontal field of view for each view (default: 90).",
    )
    parser.add_argument(
        "--view-size",
        type=int,
        default=300,
        help=(
            "Perspective view size. 300 matches the current SunCompass "
            "pre-transform centre-crop size."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metropolis_360"),
    )
    parser.add_argument(
        "--save-views",
        action="store_true",
        help="Save rendered perspective views for visual inspection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.yaw_step_deg <= 0 or args.yaw_step_deg > 360:
        raise ValueError("--yaw-step-deg must be in (0, 360]")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    yaws = np.arange(
        -180.0,
        180.0,
        args.yaw_step_deg,
        dtype=np.float64,
    ).tolist()

    if args.panorama is not None:
        records = [
            {
                "path": args.panorama,
                "token": args.panorama.stem,
                "timestamp": None,
            }
        ]
    else:
        records = find_metropolis_panoramas(
            args.metropolis_root,
            split=args.split,
            limit=args.limit,
            stride=args.stride,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = args.output_dir / "views" if args.save_views else None

    compass = SunCompass()
    compass.set_eval(dropout=False)

    all_rows = []
    summaries = []

    for index, record in enumerate(records, start=1):
        print("[{}/{}] {}".format(index, len(records), record["path"]))
        rows, summary = evaluate_panorama(
            compass=compass,
            record=record,
            yaws_deg=yaws,
            pitch_deg=args.pitch_deg,
            hfov_deg=args.hfov_deg,
            view_size=args.view_size,
            save_views_dir=views_dir,
        )
        all_rows.extend(rows)
        summaries.append(summary)

        print(
            "  consensus={:.1f} deg, median consistency error={:.1f} deg, R={:.3f}".format(
                summary["consensus_pano_azimuth_deg"],
                summary["median_abs_consistency_error_deg"],
                summary["resultant_length"],
            )
        )

    write_csv(args.output_dir / "predictions.csv", all_rows)
    write_csv(args.output_dir / "summary.csv", summaries)

    aggregate = {
        "num_panoramas": len(summaries),
        "num_views": len(all_rows),
        "median_panorama_consistency_error_deg": float(
            np.median(
                [row["median_abs_consistency_error_deg"] for row in summaries]
            )
        ),
        "mean_panorama_consistency_error_deg": float(
            np.mean(
                [row["mean_abs_consistency_error_deg"] for row in summaries]
            )
        ),
        "median_resultant_length": float(
            np.median([row["resultant_length"] for row in summaries])
        ),
    }

    with (args.output_dir / "aggregate.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2)
        handle.write("\n")

    print(json.dumps(aggregate, indent=2))
    print("Results written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
