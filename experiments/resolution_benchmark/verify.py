#!/usr/bin/env python3
"""Independently audit saved shape, training, coverage and sustained-load evidence."""
import argparse
import json
import math
from pathlib import Path
import statistics


def verify(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    assert provenance["status"] == "complete"
    assert len(results) == provenance["cases"]
    keys = [(r["size"], r["batch"], r["mode"], r["precision"], r["seconds"]) for r in results]
    assert len(keys) == len(set(keys)), "Duplicate probes"
    assert all(r["status"] in ("ok", "oom") for r in results)
    successful = [r for r in results if r["status"] == "ok"]
    for result in successful:
        assert result["actual_conv1_input"] == [result["batch"], 3, result["size"], result["size"]]
        assert result["actual_target_shape"] == [result["batch"], 2]
        assert result["gpu"] == "NVIDIA GeForce RTX 3070"
        assert result["amp_overflows"] == 0
        assert result["optimizer_updated"] == (result["mode"] != "inference")
        for key in ("first_loss", "last_loss", "images_per_second", "median_step_ms", "p95_step_ms"):
            assert math.isfinite(result[key]), (result["case"], key)
        assert result["images_per_second"] > 0
        assert result["peak_reserved_mib"] >= result["peak_allocated_mib"] > 0
        assert result["max_temp_c"] < provenance["stop_temperature_c"]
    for size in (1024, 1080):
        for mode in ("partial", "full", "inference"):
            for precision in ("fp32", "amp"):
                assert any(r["size"] == size and r["mode"] == mode and
                           r["precision"] == precision and r["batch"] == 1 for r in successful)
        assert any(r["size"] == size and r["mode"] == "partial" and
                   r["precision"] == "amp" and r["batch"] == 32 and
                   r["estimated_headroom_mib"] >= 1024 for r in successful)
    telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    soaks = [r for r in results if r["seconds"]]
    assert {(r["mode"], r["precision"]) for r in soaks} == {("partial", "fp32"), ("full", "amp")}
    sustained = []
    for result in soaks:
        assert result["status"] == "ok" and result["size"] == 1080
        assert result["elapsed_seconds"] >= 179.9 and result["steps"] >= 10
        assert result["estimated_headroom_mib"] >= 1024
        samples = [s for s in telemetry if s["case"] == result["case"]]
        assert len(samples) >= 150
        steady = [s for s in samples if s["timestamp"] >= samples[0]["timestamp"] + 10]
        count = len(steady) // 3
        first = statistics.median(s["used_mib"] for s in steady[:count])
        last = statistics.median(s["used_mib"] for s in steady[-count:])
        utilization = statistics.median(s["gpu_util_pct"] for s in steady)
        assert last - first <= 128, "Sustained device-memory growth exceeds 128 MiB"
        assert utilization >= 90, "Sustained run did not maintain substantial GPU load"
        sustained.append(dict(case=result["case"], steps=result["steps"],
                              elapsed_seconds=result["elapsed_seconds"],
                              median_gpu_utilization_pct=utilization,
                              first_third_device_memory_mib=first,
                              last_third_device_memory_mib=last,
                              memory_growth_mib=last-first, max_temp_c=result["max_temp_c"]))
    return dict(passed=True, cases=len(results), successful=len(successful),
                expected_ooms=len(results)-len(successful), sustained=sustained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.results)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
