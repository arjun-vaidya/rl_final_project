#!/usr/bin/env python3
"""Build real-data charts comparing SFT and GRPO training runs.

The script parses the existing training logs and emits:
- a CSV of parsed SFT + GRPO metrics for reproducibility
- an HTML report (using Chart.js) with real-time metrics from the runs
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


SFT_LOG_DEFAULT = Path("logs/flat_baseline_training.log")
GRPO_LOG_DEFAULT = Path("logs/final_fullpass_B120_G2_steps35.out")


def _safe_read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def _parse_dict_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None
    try:
        parsed = ast.literal_eval(line)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def parse_sft_log(log_path: Path) -> Dict:
    lines = _safe_read_lines(log_path)

    step_losses: List[Dict] = []
    summary: Dict = {}

    for line in lines:
        record = _parse_dict_line(line)
        if not record:
            continue

        if any(k.startswith("train_") for k in record.keys()):
            summary.update(record)
            continue
        if "loss" in record and "learning_rate" in record and "epoch" in record:
            step_losses.append(
                {
                    "metric_step": len(step_losses),
                    "loss": float(record["loss"]),
                    "epoch": float(record["epoch"]),
                    "learning_rate": float(record["learning_rate"]),
                }
            )

    # Infer total train steps primarily from the trainer summary.
    # Fall back to the progress line only if no summary line is available.
    total_steps = None
    for line in lines:
        if "it/s" not in line:
            continue
        m = re.search(r"\|\s*(\d+)/(\d+) \[", line)
        if m:
            total_steps = int(m.group(2))
            break

    if "train_runtime" in summary and "train_steps_per_second" in summary:
        total_steps = int(round(summary["train_runtime"] * summary["train_steps_per_second"]))
    elif total_steps is None:
        if all(k in summary for k in {"train_runtime", "train_steps_per_second"}):
            total_steps = int(round(summary["train_runtime"] * summary["train_steps_per_second"]))

    return {
        "type": "SFT",
        "log_path": str(log_path),
        "summary": summary,
        "total_steps": int(total_steps or len(step_losses)),
        "loss_records": step_losses,
    }


def parse_grpo_log(log_path: Path) -> Dict:
    lines = _safe_read_lines(log_path)

    loss_records: List[Dict] = []
    step_times: List[Dict] = []
    config: Dict = {}

    cfg_re = re.compile(r"\[train\] device=(\w+) B=(\d+) G=(\d+) reward_mode=([a-z_]+) beta=([0-9.]+)")
    step_time_re = re.compile(r"\[train\]\[step=(\d+)\] step_time_sec=([0-9.]+)")
    summary_re = re.compile(
        r"^step=\s*(\d+)\s+loss=([+-]?[0-9eE.]+)\s+outcome_acc=([0-9.]+) "
        r"router_r=([+-]?[0-9eE.]+)\s+solver_r=([+-]?[0-9eE.]+)\s+invalid_plans=(\d+)/(\d+)"
    )

    for line in lines:
        if not config and (m := cfg_re.search(line)):
            config = {
                "device": m.group(1),
                "B": int(m.group(2)),
                "G": int(m.group(3)),
                "reward_mode": m.group(4),
                "beta": float(m.group(5)),
            }

        if m := step_time_re.search(line):
            step_times.append(
                {
                    "step": int(m.group(1)),
                    "step_time_sec": float(m.group(2)),
                }
            )
            continue

        m = summary_re.search(line)
        if m:
            loss_records.append(
                {
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "outcome_acc": float(m.group(3)),
                    "router_reward": float(m.group(4)),
                    "solver_reward": float(m.group(5)),
                    "invalid_plans_num": int(m.group(6)),
                    "invalid_plans_denom": int(m.group(7)),
                }
            )

    total_steps = max([s["step"] for s in step_times] + [r["step"] for r in loss_records], default=-1) + 1

    return {
        "type": "GRPO",
        "log_path": str(log_path),
        "config": config,
        "total_steps": int(total_steps),
        "step_times": step_times,
        "loss_records": loss_records,
    }


def cumulative_from_step_times(step_times: List[Dict]) -> List[float]:
    total = 0.0
    cumulative: List[float] = []
    for row in sorted(step_times, key=lambda r: r["step"]):
        total += row["step_time_sec"]
        cumulative.append(total)
    return cumulative


def write_csv(output_path: Path, sft: Dict, grpo: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sft_rows: List[List] = []
    for item in sft["loss_records"]:
        sft_rows.append(
            ["SFT", item["metric_step"], item["loss"], item.get("epoch", ""), item.get("learning_rate", ""), ""]
        )

    for item in grpo["loss_records"]:
        sft_rows.append(
            [
                "GRPO",
                item["step"],
                item["loss"],
                "",
                "",
                item["outcome_acc"],
            ]
        )

    for idx, row in enumerate(sorted(grpo["step_times"], key=lambda r: r["step"])):
        sft_rows.append(["GRPO_STEP_TIME", row["step"], row["step_time_sec"], "", "", idx])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("pipeline,step,loss,epoch,learning_rate,outcome_accuracy\n")
        for row in sft_rows:
            line = ",".join([str(v) if v != "" else "" for v in row[:6]])
            f.write(f"{line}\n")


def write_html(output_path: Path, sft: Dict, grpo: Dict) -> None:
    total_steps = min(
        max(sft.get("total_steps", 0), 1),
        max(grpo.get("total_steps", 0), 1),
    )

    sft_total_steps = max(sft.get("total_steps", 1), 1)
    sft_runtime = float(sft["summary"].get("train_runtime", 0.0))
    sft_step_time = sft_runtime / sft_total_steps if sft_total_steps else 0.0

    grpo_step_times = sorted(grpo["step_times"], key=lambda r: r["step"])
    grpo_steps = [r["step"] for r in grpo_step_times]
    grpo_step_secs = [r["step_time_sec"] for r in grpo_step_times]
    grpo_cumulative_hours = [v / 3600.0 for v in cumulative_from_step_times(grpo_step_times)]

    x_compare = list(range(1, total_steps + 1))
    sft_cumulative_hours = [(sft_step_time * i) / 3600.0 for i in x_compare]
    sft_step_line = [sft_step_time for _ in x_compare]

    # only compare up to the available number of GRPO step-time entries
    x_step = grpo_steps
    x_step = [s + 1 for s in x_step]

    grpo_loss = [r["loss"] for r in sorted(grpo["loss_records"], key=lambda r: r["step"])]
    grpo_loss_steps = [r["step"] + 1 for r in sorted(grpo["loss_records"], key=lambda r: r["step"])]
    grpo_acc = [r["outcome_acc"] for r in sorted(grpo["loss_records"], key=lambda r: r["step"])]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Router-Solver: SFT vs GRPO Pipeline Metrics</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"></script>
  <style>
    body {{font-family: Arial, sans-serif; margin: 24px; color: #122; background: #f7f8fb;}}
    .grid {{display: grid; grid-template-columns: 1fr; gap: 28px; max-width: 1400px;}}
    canvas {{ background: #fff; border: 1px solid #d7dce2; border-radius: 8px; padding: 10px;}}
    pre {{background: #fff; border: 1px solid #d7dce2; border-radius: 8px; padding: 10px; max-width: 1400px; overflow: auto;}}
  </style>
</head>
<body>
  <h1>Router-Solver: SFT vs GRPO Training Pipeline Metrics</h1>
  <p>
    Source logs:
    <ul>
      <li><code>{sft['log_path']}</code></li>
      <li><code>{grpo['log_path']}</code></li>
    </ul>
    Parsed at generation time from raw training outputs.
  </p>
  <div class="grid">
    <div>
      <h3>Optimizer step time (seconds)</h3>
      <canvas id="stepTimeChart" height="320"></canvas>
    </div>
    <div>
      <h3>Cumulative wall-clock time (hours)</h3>
      <canvas id="cumulativeChart" height="320"></canvas>
    </div>
    <div>
      <h3>GRPO outcome trajectory</h3>
      <canvas id="qualityChart" height="320"></canvas>
    </div>
  </div>
  <script>
    const xStep = {json.dumps(x_step)};
    const grpoStepTime = {json.dumps(grpo_step_secs)};
    const xCmp = {json.dumps(x_compare)};
    const sftStepTime = {json.dumps(sft_step_line)};

    const xCmpHours = {json.dumps(x_compare)};
    const grpoCumHours = {json.dumps(grpo_cumulative_hours)};
    const sftCumHours = {json.dumps(sft_cumulative_hours)};

    const xLoss = {json.dumps(grpo_loss_steps)};
    const grpoLoss = {json.dumps(grpo_loss)};
    const grpoAcc = {json.dumps(grpo_acc)};

    const commonGrid = {{
      color: "#ddd",
      drawBorder: false,
    }};

    new Chart(document.getElementById('stepTimeChart'), {{
      type: 'line',
      data: {{
        labels: xStep,
        datasets: [
          {{label: 'GRPO step_time_sec', data: grpoStepTime, borderColor: '#0e5db8', backgroundColor: '#0e5db8', tension: 0.2, fill: false}},
          {{label: 'SFT avg step_time_sec (constant)', data: xCmp.map(() => sftStepTime[0]), borderColor: '#0ea95f', backgroundColor: '#0ea95f', borderDash: [6, 6], pointRadius: 0, fill: false}},
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{legend: {{position: 'top'}}, title: {{display: true, text: 'Per-step cost comparison (seconds)'}}}},
        scales: {{x: {{title: {{display: true, text: 'Optimizer step (1-indexed)'}}, grid: commonGrid}}, y: {{title: {{display: true, text: 'Step time (s)'}}, grid: commonGrid}}}}
      }}
    }});

    new Chart(document.getElementById('cumulativeChart'), {{
      type: 'line',
      data: {{
        labels: xCmpHours,
        datasets: [
          {{label: 'GRPO cumulative hours', data: grpoCumHours, borderColor: '#f28f1f', backgroundColor: '#f28f1f', tension: 0.25, fill: false}},
          {{label: 'SFT projected cumulative hours', data: sftCumHours, borderColor: '#0ea95f', backgroundColor: '#0ea95f', borderDash: [5,5], tension: 0.25, pointRadius: 0, fill:false}},
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{legend: {{position: 'top'}}, title: {{display: true, text: 'Cumulative elapsed training time'}}}},
        scales: {{x: {{title: {{display: true, text: 'Optimizer steps'}}, grid: commonGrid}}, y: {{title: {{display: true, text: 'Hours'}}, grid: commonGrid}}}}
      }}
    }});

    new Chart(document.getElementById('qualityChart'), {{
      type: 'line',
      data: {{
        labels: xLoss,
        datasets: [
          {{
            label: 'GRPO outcome_acc',
            data: grpoAcc,
            borderColor: '#7b2cbf',
            backgroundColor: '#7b2cbf',
            yAxisID: 'y2',
            tension: 0.2,
            fill: false,
          }},
          {{
            label: 'GRPO loss',
            data: grpoLoss,
            borderColor: '#0e5db8',
            backgroundColor: '#0e5db8',
            yAxisID: 'y',
            tension: 0.2,
            fill: false,
          }},
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{legend: {{position: 'top'}}, title: {{display: true, text: 'GRPO quality trajectory (logged checkpoints)'}}}},
        scales: {{
          x: {{title: {{display: true, text: 'Optimizer step'}}, grid: commonGrid}},
          y: {{type: 'linear', position: 'left', title: {{display: true, text: 'GRPO loss'}}, grid: commonGrid}},
          y2: {{type: 'linear', position: 'right', title: {{display: true, text: 'outcome_acc'}}, grid: {{drawOnChartArea: false}}}},
        }}
      }}
    }});
  </script>
  <pre>
Source summary:
{json.dumps({
  "sft_total_steps": sft.get("total_steps"),
  "sft_train_runtime_sec": round(sft_runtime, 2),
  "sft_avg_step_time_sec": round(sft_step_time, 4) if sft_step_time else None,
  "grpo_total_steps": grpo.get("total_steps"),
  "grpo_avg_step_time_sec": round(sum(grpo_step_secs)/max(len(grpo_step_secs),1), 4),
  "grpo_final_outcome_acc": grpo_loss_records_last_loss(grpo),
}, indent=2)}
  </pre>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)


def _to_int(v: float, lo: float, hi: float, px0: float, px1: float) -> int:
    if hi == lo:
        return int((px0 + px1) / 2)
    ratio = (v - lo) / (hi - lo)
    return int(px1 - ratio * (px1 - px0))


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    bbox: tuple,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    font: ImageFont.FreeTypeFont,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#333333",
) -> None:
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=color)

    tick_color = "#bbbbbb"
    for i in range(1, 6):
        tx = x0 + (x1 - x0) * i / 5
        ty = y1
        draw.line([(tx, y0), (tx, y1)], fill=tick_color, width=1)
        x_val = x_min + (x_max - x_min) * i / 5
        draw.text((tx - 10, y1 + 4), f"{x_val:.0f}", fill=color, font=font)

    for i in range(1, 6):
        ty = y0 + (y1 - y0) * i / 5
        draw.line([(x0, ty), (x1, ty)], fill=tick_color, width=1)
        y_val = y_min + (y_max - y_min) * (1 - i / 5)
        draw.text((x0 - 52, ty - 6), f"{y_val:.2f}", fill=color, font=font)

    draw.text((x0 + (x1 - x0) // 2 - len(title) * 3, y0 - 26), title, fill=color, font=font)
    draw.text((x0 + (x1 - x0) // 2 - len(x_label) * 3, y1 + 22), x_label, fill=color, font=font)
    draw.text((x0 - 60, y0 + 10), y_label, fill=color, font=font)


def _draw_line(
    draw: ImageDraw.ImageDraw,
    xs: List[float],
    ys: List[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bbox: tuple,
    color: str,
) -> None:
    x0, y0, x1, y1 = bbox
    if len(xs) != len(ys) or not xs:
        return
    points: List[tuple] = []
    for xv, yv in zip(xs, ys):
        x_pix = int(x0 + (xv - x_min) / (x_max - x_min) * (x1 - x0) if x_max != x_min else (x0 + x1) / 2)
        y_pix = _to_int(yv, y_min, y_max, y0, y1)
        points.append((x_pix, y_pix))

    if len(points) > 1:
        draw.line(points, fill=color, width=3)
    for p in points:
        draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=color, outline=color)


def write_png(output_path: Path, sft: Dict, grpo: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 3600
    height = 3000
    margin = 120
    panel_h = 780
    gap = 110
    panel_left = 220
    panel_right = width - 120
    plot_h = 520

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except Exception:
        font = ImageFont.load_default()

    draw.text((margin, 40), "Router-Solver: SFT vs GRPO Training Pipeline Metrics", fill="#111111", font=font)

    # 1) step time
    grpo_step_times = sorted(grpo["step_times"], key=lambda r: r["step"])
    x_step = [r["step"] + 1 for r in grpo_step_times]
    y_step = [r["step_time_sec"] for r in grpo_step_times]
    sft_step_time = 0.0
    if (st := sft.get("summary", {}).get("train_runtime", 0.0)) and sft.get("total_steps"):
        sft_step_time = st / max(1, int(sft["total_steps"]))

    if x_step:
        x_min, x_max = min(x_step), max(x_step)
        x_max = max(1, x_max)
        y_min = min(min(y_step), sft_step_time)
        y_max = max(max(y_step), sft_step_time)
        y_min = min(y_min, 0.0)
        y_max = y_max * 1.08 + 1e-6

        y0 = 90
        y1 = y0 + plot_h
        x0 = panel_left
        x1 = panel_right
        _draw_axes(draw, (x0, y0, x1, y1), x_min, x_max, y_min, y_max, font, "Optimizer step time (s)", "Step", "seconds", color="#111111")
        _draw_line(draw, x_step, y_step, x_min, x_max, y_min, y_max, (x0, y0, x1, y1), "#0e5db8")
        _draw_line(draw, x_step, [sft_step_time for _ in x_step], x_min, x_max, y_min, y_max, (x0, y0, x1, y1), "#0ea95f")

    # 2) cumulative hours
    grpo_cum = [t / 3600.0 for t in cumulative_from_step_times(grpo_step_times)]
    x_cmp = list(range(1, len(grpo_cum) + 1))
    if not x_cmp:
        x_cmp = []

    sft_runtime = float(sft["summary"].get("train_runtime", 0.0))
    sft_total_steps = max(1, int(sft.get("total_steps", 1)))
    sft_avg_step = sft_runtime / sft_total_steps if sft_total_steps else 0.0
    sft_cum = [(sft_avg_step * i) / 3600.0 for i in x_cmp]

    if x_cmp:
        y_min = min(min(grpo_cum), min(sft_cum))
        y_max = max(max(grpo_cum), max(sft_cum))
        y_min = min(y_min, 0.0)
        y_max = y_max * 1.08 + 1e-6
        y0 = 90 + panel_h + gap
        y1 = y0 + plot_h
        x0 = panel_left
        x1 = panel_right
        _draw_axes(
            draw,
            (x0, y0, x1, y1),
            min(x_cmp),
            max(x_cmp),
            y_min,
            y_max,
            font,
            "Cumulative elapsed hours",
            "Step",
            "Hours",
            color="#111111",
        )
        _draw_line(draw, [float(v) for v in x_cmp], grpo_cum, min(x_cmp), max(x_cmp), y_min, y_max, (x0, y0, x1, y1), "#f28f1f")
        _draw_line(draw, [float(v) for v in x_cmp], sft_cum, min(x_cmp), max(x_cmp), y_min, y_max, (x0, y0, x1, y1), "#0ea95f")

    # 3) outcome and GRPO loss
    loss_rows = sorted(grpo["loss_records"], key=lambda r: r["step"])
    x_loss = [float(r["step"] + 1) for r in loss_rows]
    y_loss = [float(r["loss"]) for r in loss_rows]
    y_acc = [float(r["outcome_acc"]) for r in loss_rows]

    if x_loss:
        x_min, x_max = min(x_loss), max(x_loss)
        y_min = min(y_loss)
        y_max = max(y_loss)
        y_span = y_max - y_min
        if y_span == 0:
            y_min -= 1
            y_max += 1
        else:
            y_min -= 0.1 * y_span
            y_max += 0.1 * y_span

        acc_min = 0.0
        acc_max = max(1.0, max(y_acc) if y_acc else 1.0)
        y0 = 90 + 2 * (panel_h + gap)
        y1 = y0 + plot_h
        x0 = panel_left
        x1 = panel_right

        _draw_axes(
            draw,
            (x0, y0, x1, y1),
            x_min,
            x_max,
            y_min,
            y_max,
            font,
            "GRPO loss and outcome_acc",
            "Step",
            "Loss",
            color="#111111",
        )
        _draw_line(draw, x_loss, y_loss, x_min, x_max, y_min, y_max, (x0, y0, x1, y1), "#0e5db8")

        # overlay outcome_acc mapped onto the same vertical scale as loss
        acc_scale = (y_max - y_min) / (acc_max - acc_min if acc_max != acc_min else 1)
        y_acc_scaled = [y_max - (v - acc_min) * acc_scale for v in y_acc]
        _draw_line(
            draw,
            x_loss,
            y_acc_scaled,
            x_min,
            x_max,
            y_min,
            y_max,
            (x0, y0, x1, y1),
            "#7b2cbf",
        )

    with output_path.open("wb") as f:
        img.save(f, format="PNG")

    print(f"Wrote {output_path}")


def grpo_loss_records_last_loss(grpo: Dict) -> Dict:
    if not grpo["loss_records"]:
        return {}
    last = sorted(grpo["loss_records"], key=lambda r: r["step"])[-1]
    return {
        "step": last["step"],
        "loss": last["loss"],
        "outcome_acc": last["outcome_acc"],
        "invalid_plans": f"{last['invalid_plans_num']}/{last['invalid_plans_denom']}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft-log",
        type=Path,
        default=SFT_LOG_DEFAULT,
        help="Path to flat SFT log file.",
    )
    parser.add_argument(
        "--grpo-log",
        type=Path,
        default=GRPO_LOG_DEFAULT,
        help="Path to GRPO log file (typically full-pass run).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Directory for generated chart artifacts.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV with merged parsed metrics.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Emit a PNG chart image from real parsed logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    sft_log = (repo_root / args.sft_log).resolve() if not args.sft_log.is_absolute() else args.sft_log
    grpo_log = (repo_root / args.grpo_log).resolve() if not args.grpo_log.is_absolute() else args.grpo_log

    if not sft_log.exists():
        raise FileNotFoundError(f"SFT log not found: {sft_log}")
    if not grpo_log.exists():
        raise FileNotFoundError(f"GRPO log not found: {grpo_log}")

    sft_data = parse_sft_log(sft_log)
    grpo_data = parse_grpo_log(grpo_log)

    output_dir = (repo_root / args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_pipeline_data.csv"
    html_path = output_dir / "training_pipeline_chart.html"

    if args.csv:
        write_csv(csv_path, sft_data, grpo_data)
        print(f"Wrote {csv_path}")
    else:
        if csv_path.exists():
            csv_path.unlink()

    if args.png:
        png_path = output_dir / "training_pipeline_chart.png"
        write_png(png_path, sft_data, grpo_data)

    write_html(html_path, sft_data, grpo_data)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
