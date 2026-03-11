#!/usr/bin/env python3
"""
Plot and summarize gradient-sensitivity logs.

This script always writes CSV summaries from grad_sensitivity.jsonl.
If matplotlib is available, it also writes PNG line plots.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PLOT_SERIES = {
    "lora_rel": [
        "grad_sens/lora_A_text_relative_grad_norm",
        "grad_sens/lora_A_visual_relative_grad_norm",
        "grad_sens/lora_A_audio_relative_grad_norm",
        "grad_sens/lora_B_shared_relative_grad_norm",
    ],
    "lora_grad": [
        "grad_sens/lora_A_text_grad_norm",
        "grad_sens/lora_A_visual_grad_norm",
        "grad_sens/lora_A_audio_grad_norm",
        "grad_sens/lora_B_shared_grad_norm",
    ],
    "projector_rel": [
        "grad_sens/vl_projector_relative_grad_norm",
        "grad_sens/al_projector_relative_grad_norm",
    ],
}


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    running = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def _load_rows(jsonl_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["step"]))
    return rows


def _write_long_csv(rows: Sequence[Dict], out_csv: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys() if k.startswith("grad_sens/")})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "metric", "value"])
        for r in rows:
            step = int(r["step"])
            epoch = float(r.get("epoch", 0.0))
            for k in keys:
                if k in r:
                    w.writerow([step, epoch, k, float(r[k])])


def _write_summary_csv(rows: Sequence[Dict], out_csv: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys() if k.startswith("grad_sens/")})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "min", "max"])
        for k in keys:
            vals = [float(r[k]) for r in rows if k in r]
            if not vals:
                continue
            w.writerow([k, len(vals), sum(vals) / len(vals), min(vals), max(vals)])


def _maybe_plot(rows: Sequence[Dict], output_dir: Path, smooth_window: int) -> Tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False, "matplotlib not installed; wrote CSV files only."

    steps = [int(r["step"]) for r in rows]
    for group_name, metrics in PLOT_SERIES.items():
        plt.figure(figsize=(10, 5))
        for metric in metrics:
            vals = [float(r.get(metric, 0.0)) for r in rows]
            vals = _moving_average(vals, smooth_window)
            short_label = metric.replace("grad_sens/", "")
            plt.plot(steps, vals, label=short_label, linewidth=1.6)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"{group_name} (smooth={smooth_window})")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{group_name}.png", dpi=180)
        plt.close()
    return True, "PNG plots written."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to grad_sensitivity.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output directory for CSV/plots")
    parser.add_argument("--smooth_window", type=int, default=10, help="Moving-average window for plots")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(jsonl_path)
    if not rows:
        raise SystemExit(f"No rows found in {jsonl_path}")

    _write_long_csv(rows, out_dir / "grad_sensitivity_long.csv")
    _write_summary_csv(rows, out_dir / "grad_sensitivity_summary.csv")
    plotted, msg = _maybe_plot(rows, out_dir, max(1, args.smooth_window))

    print(f"Rows: {len(rows)}")
    print(f"Step range: {rows[0]['step']} -> {rows[-1]['step']}")
    print(f"Wrote: {out_dir / 'grad_sensitivity_long.csv'}")
    print(f"Wrote: {out_dir / 'grad_sensitivity_summary.csv'}")
    print(msg)
    if not plotted:
        print("Install matplotlib to get PNGs: pip install matplotlib")


if __name__ == "__main__":
    main()

