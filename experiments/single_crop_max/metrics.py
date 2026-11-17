import numpy as np
import math

def summarize_comparison(records):
    # Input validation
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("records must be a non-empty list")

    scene_ids = set()
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("Each record must be a dict")

        # Check required keys
        required_keys = {"scene_id", "timestamps", "single_errors", "max_errors", "single_seconds", "max_seconds"}
        if set(rec.keys()) != required_keys:
            raise ValueError(f"Record missing required keys: {required_keys - set(rec.keys())}")

        scene_id = rec["scene_id"]
        if not isinstance(scene_id, str):
            raise ValueError("scene_id must be a string")
        if scene_id in scene_ids:
            raise ValueError(f"Duplicate scene_id: {scene_id}")
        scene_ids.add(scene_id)

        timestamps = rec["timestamps"]
        if not isinstance(timestamps, int) or timestamps <= 0:
            raise ValueError("timestamps must be a positive integer")

        single_errors = rec["single_errors"]
        max_errors = rec["max_errors"]
        if not isinstance(single_errors, list) or not isinstance(max_errors, list):
            raise ValueError("single_errors and max_errors must be lists")

        expected_len = timestamps * 10
        if len(single_errors) != expected_len or len(max_errors) != expected_len:
            raise ValueError(f"Error lists must have length timestamps*10 = {expected_len}")

        # Validate errors
        for err in single_errors + max_errors:
            if err is not None:
                if not isinstance(err, (int, float)) or not math.isfinite(err) or err < -180 or err >= 180:
                    raise ValueError("All errors must be finite numbers in [-180, 180) or None")

        # Validate durations
        single_seconds = rec["single_seconds"]
        max_seconds = rec["max_seconds"]
        if not isinstance(single_seconds, (int, float)) or not math.isfinite(single_seconds) or single_seconds <= 0:
            raise ValueError("single_seconds must be a finite positive number")
        if not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be a finite positive number")

    # Helper function to compute metrics
    def compute_metrics(errors):
        # Filter valid errors
        valid_errors = [e for e in errors if e is not None]
        total = len(errors)
        valid = len(valid_errors)
        invalid = total - valid

        if valid == 0:
            return {
                "total": total,
                "valid": valid,
                "invalid": invalid,
                "mae_deg": None,
                "median_deg": None,
                "rmse_deg": None,
                "p90_deg": None,
                "p95_deg": None,
                "max_deg": None,
                "circular_bias_deg": None,
                "within_5": None,
                "within_10": None,
                "within_20": None
            }

        # Convert to numpy array
        arr = np.array(valid_errors, dtype=np.float64)

        # Basic stats
        mae = np.mean(np.abs(arr))
        median = np.median(np.abs(arr))
        rmse = np.sqrt(np.mean(arr**2))
        max_val = np.max(np.abs(arr))

        # Percentiles using linear interpolation (default numpy)
        p90 = np.percentile(np.abs(arr), 90)
        p95 = np.percentile(np.abs(arr), 95)

        # Circular bias: atan2(mean sin, mean cos)
        rad = np.deg2rad(arr)
        sin_mean = np.mean(np.sin(rad))
        cos_mean = np.mean(np.cos(rad))
        circular_bias = np.rad2deg(np.arctan2(sin_mean, cos_mean))

        # Within thresholds
        within_5 = np.sum(np.abs(arr) <= 5) / valid
        within_10 = np.sum(np.abs(arr) <= 10) / valid
        within_20 = np.sum(np.abs(arr) <= 20) / valid

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "mae_deg": float(mae),
            "median_deg": float(median),
            "rmse_deg": float(rmse),
            "p90_deg": float(p90),
            "p95_deg": float(p95),
            "max_deg": float(max_val),
            "circular_bias_deg": float(circular_bias),
            "within_5": float(within_5),
            "within_10": float(within_10),
            "within_20": float(within_20)
        }

    # Process each scene
    scene_results = []
    single_errors_all = []
    max_errors_all = []
    single_seconds_total = 0.0
    max_seconds_total = 0.0
    improved_count = 0
    tied_count = 0
    worse_count = 0
    uncomparable_count = 0

    for rec in records:
        scene_id = rec["scene_id"]
        single_errors = rec["single_errors"]
        max_errors = rec["max_errors"]
        single_seconds = rec["single_seconds"]
        max_seconds = rec["max_seconds"]

        # Compute metrics for this scene
        single_metrics = compute_metrics(single_errors)
        max_metrics = compute_metrics(max_errors)

        # Compute MAE change
        single_mae = single_metrics["mae_deg"]
        max_mae = max_metrics["mae_deg"]

        if single_mae is None or max_mae is None:
            mae_change = None
            uncomparable_count += 1
        else:
            mae_change = max_mae - single_mae
            if mae_change < 0:
                improved_count += 1
            elif mae_change == 0:
                tied_count += 1
            else:
                worse_count += 1

        scene_results.append({
            "scene_id": scene_id,
            "single": single_metrics,
            "single-max": max_metrics,
            "mae_change_deg": mae_change
        })

        # Accumulate for pooled stats
        single_errors_all.extend(single_errors)  # Keep None to preserve invalid count
        max_errors_all.extend(max_errors)  # Keep None to preserve invalid count
        single_seconds_total += single_seconds
        max_seconds_total += max_seconds

    # Compute pooled metrics
    pooled_single = compute_metrics(single_errors_all)
    pooled_max = compute_metrics(max_errors_all)

    # Compute totals
    scene_count = len(records)
    total_timestamps = sum(rec["timestamps"] for rec in records)
    timestamps_per_method = total_timestamps
    predictions_per_method = 5 * total_timestamps
    pairs_per_method = 10 * total_timestamps

    return {
        "scene_count": scene_count,
        "timestamps_per_method": timestamps_per_method,
        "predictions_per_method": predictions_per_method,
        "pairs_per_method": pairs_per_method,
        "pooled": {
            "single": pooled_single,
            "single-max": pooled_max
        },
        "by_scene": scene_results,
        "improved_scene_count": improved_count,
        "tied_scene_count": tied_count,
        "worse_scene_count": worse_count,
        "uncomparable_scene_count": uncomparable_count,
        "processing_seconds": {
            "single": float(single_seconds_total),
            "single-max": float(max_seconds_total)
        }
    }
