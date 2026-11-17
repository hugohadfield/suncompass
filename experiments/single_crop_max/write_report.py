"""Write a factual report from independently checked numerical outputs."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    data=json.loads((root/'comparison/analysis.json').read_text())
    check=json.loads((root/'comparison/verification.json').read_text());assert check['passed']
    single=data['pooled']['single'];maximum=data['pooled']['single-max'];letterbox=data['historical_letterbox_pooled']
    diff=maximum['mae_deg']-single['mae_deg'];percent=100*(single['mae_deg']-maximum['mae_deg'])/single['mae_deg']
    scenes=data['by_scene'];six=next(r for r in scenes if r['scene_id']=='scene_006')
    provenance=data['provenance'][0]
    lines=['# Single Crop Max versus OG single crop — Sunny63','',
      f"Single Crop Max pooled relative-yaw MAE is **{maximum['mae_deg']:.2f}°**, versus **{single['mae_deg']:.2f}°** for OG single crop ({diff:+.2f}°, {percent:.1f}% lower). It improves scene MAE in **{data['improved_scene_count']}/63 scenes**; {data['worse_scene_count']} worsen and {data['tied_scene_count']} tie.",'',
      'The same frozen checkpoint was evaluated on the exact original 63 Sunny scenes, including the Sunny/Dusk scene_055. Both methods cover 12,606 synchronized timestamps, 63,030 images and 126,060 camera pairs. OG predictions are reused from their preserved databases.','',
      '| Method | MAE ° | Median ° | RMSE ° | p95 ° | Within 10° | Valid / total pairs |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in [('OG single crop',single),('Single Crop Max',maximum),('Letterbox (historical context)',letterbox)]:
        lines.append(f"| {name} | {m['mae_deg']:.2f} | {m['median_deg']:.2f} | {m['rmse_deg']:.2f} | {m['p95_deg']:.2f} | {m['within_10']:.2%} | {m['valid']:,} / {m['total']:,} |")
    lines+=['',f"New invalid predictions: {sum(r['invalid_predictions'] for r in data['provenance'])}; invalid pairs: {maximum['invalid']}. Historical single metrics were recomputed from the baseline databases and matched the preserved Sunny63 report. Letterbox numbers above are historical context, not a fresh run.",'',
      '## What changed','',
      'Single Crop Max keeps the largest centered square from the original RGB image, then resizes it directly to 224×224 with Pillow bilinear. A 1920×1080 input supplies a 1080×1080 crop at (420, 0, 1500, 1080), rather than the original 300×300 crop. Integer origins use floor division, with odd excess removed from the right/bottom. ImageNet normalization runs once; the internal ResNet resize/crop is bypassed. There is one model pass per image and no retraining or weight change.','',
      'The original single, scan and letterbox interfaces are preserved. Select this method with `--method single-max` or `SunCompass.single_max_predict(image)`.','',
      '## Scene 006 and synchronized visualization','',
      f"Scene 006 MAE: **{six['single-max']['mae_deg']:.2f}°** versus OG **{six['single']['mae_deg']:.2f}°**. Both use the same 200 timestamps, 1,000 images and 2,000 pairs.",'',
      '[Five-camera GIF](../media/scene_006-single-max-five-cameras.gif) · [MP4](../media/scene_006-single-max-five-cameras.mp4)','',
      'The movie preserves the accepted 1584×1132 layout, camera order, fonts, colors, direction dials and shared yaw-aligned panel. It replays every other source frame: 100 synchronized frames at 5 fps, a 20-second loop. The gold outline and input preview show the maximum square crop. Preview pixels were checked against the denormalized production network input; the movie replays saved predictions. No image-plane solar location is inferred.','',
      '## Processing time','']
    seconds=data['processing_seconds']
    lines+=[f"Summed measured timestamp processing: OG {seconds['single']:.2f} s; Single Crop Max {seconds['single-max']:.2f} s ({seconds['single-max']/seconds['single']:.2f}×). Average per five-camera timestamp: {seconds['single']/12606:.4f} s and {seconds['single-max']/12606:.4f} s respectively.",'',
      'These are instrumented processing times including image decoding and inference, excluding warmup, data staging and some overhead. The comparison uses historical OG timing and one new run; caching and machine state may differ. It is not a controlled latency benchmark or a timing confidence interval.','',
      '## Validation and provenance','',
      f"Inference and rendering source: `{provenance['code_revision']}`. Full baseline revisions are retained per scene in `analysis.json`. Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`.",'',
      'Every scene passed SQLite integrity/foreign-key checks, positive timestamp and 5-image/10-pair coverage, exact scene/run/method/subset checks, identical manifests, input hashes, checkpoint, package environment, GPU identity, pair keys and COLMAP references. Wrapped estimates/errors were independently recalculated from named camera predictions. Numerical aggregation was cross-checked against the maintained pilot metrics implementation. All 35 implementation tests passed on PC2, including CUDA model contracts and preview parity; none were skipped.','',
      'Local QWEN authored candidates for inference, rendering and aggregation through tool-free bounded requests. Seven attempts across three tasks were retained. First-pass success was 0/3; inference and rendering required supervisor completion, while aggregation passed after one QWEN repair. This is supervisor-assisted work. Local API usage is recorded in the audit; frontier usage and token savings are not measured.','',
      '## Interpretation','',
      'Errors measure inter-camera relative-yaw disagreement against COLMAP camera orientation, not absolute sun-bearing error or surveyed heading. Frames and pairs are correlated; these 63 previously examined Sunny scenes are not a new held-out generalization test. Fisheye distortion, occlusions and camera tilt remain. This checkpoint was trained with a different crop distribution. Substantial residual error remains even where this preprocessing improves the comparison.','',
      '## Per-scene results','',
      '| Scene | OG MAE ° | Single Crop Max MAE ° | Change ° |','|---|---:|---:|---:|']
    for r in scenes:lines.append(f"| {r['scene_id']} | {r['single']['mae_deg']:.2f} | {r['single-max']['mae_deg']:.2f} | {r['mae_change_deg']:+.2f} |")
    lines+=['','![Error distributions and per-scene comparison](comparison.png)','','Artifacts: `analysis.json`, `verification.json`, `per-scene.csv`, `pair-results.csv`, and `scene006-predictions.csv`. Per-scene raw databases, configurations, logs and checksummed backups are preserved alongside the comparison. See the experiment record for backup manifests and exact regeneration commands.','']
    (root/'comparison/report.md').write_text('\n'.join(lines));print(root/'comparison/report.md')


if __name__=='__main__':main()
