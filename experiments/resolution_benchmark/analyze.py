#!/usr/bin/env python3
"""Summarize saved capacity probes without rerunning CUDA work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def summarize(results):
    groups = {}
    for result in results:
        if result["seconds"]:
            continue
        key = (result["size"], result["mode"], result["precision"])
        groups.setdefault(key, []).append(result)
    capacities = []
    for (size, mode, precision), cases in sorted(groups.items()):
        fits = [r for r in cases if r["status"] == "ok"]
        safe = [r for r in fits if r["estimated_headroom_mib"] >= 1024]
        ooms = [r["batch"] for r in cases if r["status"] == "oom"]
        entry = dict(size=size, mode=mode, precision=precision,
                     max_tested_fit=max((r["batch"] for r in fits), default=None),
                     smallest_tested_oom=min(ooms, default=None))
        if safe:
            best = max(safe, key=lambda r: r["batch"])
            entry.update(safe_tested_batch=best["batch"],
                         safe_images_per_second=best["images_per_second"],
                         safe_peak_reserved_mib=best["peak_reserved_mib"],
                         safe_estimated_headroom_mib=best["estimated_headroom_mib"])
        capacities.append(entry)
    soaks = [r for r in results if r["seconds"]]
    return dict(case_count=len(results), successes=sum(r["status"] == "ok" for r in results),
                expected_ooms=sum(r["status"] == "oom" for r in results),
                errors=[r for r in results if r["status"] not in ("ok", "oom")],
                max_temp_c=max(r["max_temp_c"] for r in results),
                max_power_w=max(r["max_power_w"] for r in results),
                capacities=capacities, soaks=soaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.results / "results.json").read_text())
    provenance = json.loads((args.results / "provenance.json").read_text())
    summary = summarize(results)
    summary["provenance"] = provenance
    summary["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["analysis_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# PC2 SunCompass GPU resolution benchmark", "",
             f"Source: `{provenance['code_revision']}`; Slurm job `{provenance['slurm_job_id']}`.",
             f"Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`.", "",
             f"{summary['case_count']} cases: {summary['successes']} successful, "
             f"{summary['expected_ooms']} expected OOMs, {len(summary['errors'])} errors. "
             f"Maximum sampled GPU temperature {summary['max_temp_c']:.0f}°C; "
             f"power {summary['max_power_w']:.1f} W.", "",
             "## Requested training resolutions", "",
             "Safe means a tested batch with estimated ≥1 GiB VRAM headroom. "
             "A fit at batch 32 is a lower bound on capacity. "
             "Throughput excludes input I/O, decoding, augmentation, transfers and checkpoints.", "",
             "| Input | Training | Precision | Largest fit | First OOM | Safe tested batch | Images/s | Peak reserved GiB | Headroom GiB |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    reused = [r for r in results if r.get("reused_from")]
    if reused:
        previous = provenance["reused_provenance"]
        lines[6:6] = ["", f"{len(reused)} completed probes were retained from Slurm job "
                      f"`{previous['slurm_job_id']}` at source `{previous['code_revision']}`. "
                      "The compute contract is identical; the continuation adds explicit "
                      "CUDA-library allocation-failure classification. Individual records "
                      "retain their original revision and artifact path.", ""]
    for entry in summary["capacities"]:
        if entry["size"] not in (512, 768, 1024, 1080) or entry["mode"] == "inference":
            continue
        safe = entry.get("safe_tested_batch")
        lines.append(f"| {entry['size']}² | {entry['mode']} | {entry['precision']} | "
                     f"{entry['max_tested_fit']} | {entry['smallest_tested_oom'] or '—'} | "
                     f"{safe or '—'} | " + (f"{entry['safe_images_per_second']:.1f} | "
                     f"{entry['safe_peak_reserved_mib']/1024:.2f} | "
                     f"{entry['safe_estimated_headroom_mib']/1024:.2f} |" if safe else "— | — | — |"))
    lines += ["", "## Sustained 1080 training", ""]
    for soak in summary["soaks"]:
        lines.append(f"- {soak['mode']}/{soak['precision']}, batch {soak['batch']}: "
                     f"{soak['steps']} steps over {soak['elapsed_seconds']:.1f} s, "
                     f"{soak['images_per_second']:.1f} images/s, "
                     f"{soak['peak_reserved_mib']/1024:.2f} GiB peak reserved, "
                     f"{soak['estimated_headroom_mib']/1024:.2f} GiB estimated headroom, "
                     f"{soak['max_temp_c']:.0f}°C max; {soak['amp_overflows']} AMP overflows.")
    lines += ["", "## Batch-one resolution envelope", "",
              "These are tested bounds, not theoretical limits. No activation checkpointing, "
              "gradient accumulation, compilation or alternative memory format was used.", ""]
    for mode in ("partial", "full"):
        for precision in ("fp32", "amp"):
            cases = [r for r in results if r["mode"] == mode and r["precision"] == precision
                     and r["batch"] == 1 and not r["seconds"]]
            fits = [r["size"] for r in cases if r["status"] == "ok"]
            ooms = [r["size"] for r in cases if r["status"] == "oom"]
            lines.append(f"- {mode}/{precision}: largest tested fit {max(fits, default=None)}²; "
                         f"first tested OOM {min(ooms) if ooms else 'not reached'}.")
    lines += ["", "## Scope and interpretation", "",
              "The original loader's RandomCrop(400) is not the network input limit: "
              "the model's own default preprocessing crops to 224. Every successful probe "
              "asserts its real conv1 input shape and bypasses that preprocessing.", "",
              "Synthetic 1024×512 RGB fixtures are independently generated per batch member, "
              "letterboxed with centered padding, and normalized once. All image pixels and "
              "padding are processed densely. Two targets represent forward/left components. "
              "This validates compute capacity and finite optimizer updates, not label semantics, "
              "training convergence or accuracy. The AMOS importer/exporter and production "
              "preprocessing still need separate implementation.", "",
              "Partial fine-tuning retains the existing trainable layer3/layer4/classifier/head "
              "configuration. Full fine-tuning trains every parameter. Both use Adam/MSE. "
              "FP32 and FP16 AMP are separate profiles. Timed steps synchronize CUDA and "
              "check finite outputs/loss; warmup initializes optimizer state. Each case runs "
              "in a fresh process, with expected OOMs preserved. Headroom combines peak "
              "PyTorch reservation (including warmup) and end-of-case external device use. "
              "Temperature, power, utilization and device memory are sampled every second.", "",
              "The CPU input pipeline and a full validation/checkpoint cycle can use additional "
              "resources; begin a real-data smoke run at a conservative batch before choosing "
              "the final training configuration. No production checkpoint was updated.", "",
              f"Analysis source: `{summary['analysis_revision']}`, script SHA-256 "
              f"`{summary['analysis_sha256']}`.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("capacities", "soaks", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
