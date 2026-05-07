#!/usr/bin/env python3
"""Create a compact training-results chart from a WandB output log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STEP_RE = re.compile(
    r"^step=\s*(\d+)\s+loss=([+-]?[0-9eE.]+)\s+outcome_acc=([0-9.]+)\s+"
    r"router_r=([+-]?[0-9eE.]+)\s+solver_r=([+-]?[0-9eE.]+)\s+"
    r"invalid_plans=(\d+)/(\d+)"
)
TIME_RE = re.compile(r"\[train\]\[step=(\d+)\] step_time_sec=([0-9.]+)")
CFG_RE = re.compile(r"\[train\] device=(\w+) B=(\d+) G=(\d+) reward_mode=([a-z_]+) beta=([0-9.]+)")


def parse_log(path: Path):
    steps = []
    losses = []
    accs = []
    router_rewards = []
    solver_rewards = []
    invalid = []
    invalid_total = []
    times = []
    cfg = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if cfg is None:
            m = CFG_RE.search(line)
            if m:
                cfg = {
                    "device": m.group(1),
                    "B": int(m.group(2)),
                    "G": int(m.group(3)),
                    "reward_mode": m.group(4),
                    "beta": float(m.group(5)),
                }

        m = STEP_RE.search(line)
        if m:
            s = int(m.group(1))
            while len(steps) <= s:
                steps.append(len(steps))
                losses.append(None)
                accs.append(None)
                router_rewards.append(None)
                solver_rewards.append(None)
                invalid.append(None)
                invalid_total.append(1)
            losses[s] = float(m.group(2))
            accs[s] = float(m.group(3))
            router_rewards[s] = float(m.group(4))
            solver_rewards[s] = float(m.group(5))
            invalid[s] = int(m.group(6))
            invalid_total[s] = max(1, int(m.group(7)))

        m = TIME_RE.search(line)
        if m:
            s = int(m.group(1))
            while len(times) <= s:
                times.append(0.0)
            times[s] = float(m.group(2))

    if cfg is None:
        cfg = {"device": "unknown", "B": 0, "G": 0, "reward_mode": "unknown", "beta": 0.0}

    return {
        "cfg": cfg,
        "steps": steps,
        "losses": losses,
        "accuracy": accs,
        "router_rewards": router_rewards,
        "solver_rewards": solver_rewards,
        "invalid": invalid,
        "invalid_total": invalid_total,
        "step_times": times,
    }


def linmap(v, v0, v1, t0, t1):
    if v1 == v0:
        return (t0 + t1) / 2
    return t0 + (v - v0) * (t1 - t0) / (v1 - v0)


def draw_series(draw, bbox, xs, ys, xlim, ylim, color, width=5):
    x0, y0, x1, y1 = bbox
    if len(xs) < 2:
        return
    for a, b in zip(range(len(xs) - 1), range(1, len(xs))):
        if ys[a] is None or ys[b] is None:
            continue
        x_a = x0 + linmap(xs[a], xlim[0], xlim[1], 0, x1 - x0)
        x_b = x0 + linmap(xs[b], xlim[0], xlim[1], 0, x1 - x0)
        y_a = y1 - linmap(ys[a], ylim[0], ylim[1], 0, y1 - y0)
        y_b = y1 - linmap(ys[b], ylim[0], ylim[1], 0, y1 - y0)
        draw.line((x_a, y_a, x_b, y_b), fill=color, width=width)
        r = width * 2
        draw.ellipse((x_a - r, y_a - r, x_a + r, y_a + r), fill=color, outline=color)
    if len(xs) > 0 and ys:
        idx = len(xs) - 1
        if ys[idx] is not None:
            x_last = x0 + linmap(xs[idx], xlim[0], xlim[1], 0, x1 - x0)
            y_last = y1 - linmap(ys[idx], ylim[0], ylim[1], 0, y1 - y0)
            r = width * 2
            draw.ellipse((x_last - r, y_last - r, x_last + r, y_last + r), fill=color, outline=color)


def draw_axis(draw, bbox, x_min, x_max, y_min, y_max, title, ylabel, font):
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1, y1), outline="#222222", width=2)
    # remove in-chart text for legibility at slide scale
    _ = (title, ylabel, font, x_min, x_max, y_min, y_max)


def render_panel(out_png: Path, title: str, y_values, x_values=None, color="#3366cc", font=None, small_font=None):
    if x_values is None:
        x_values = list(range(len(y_values)))
    finite_y = [y for y in y_values if y is not None]
    if not finite_y:
        finite_y = [0.0]

    y_min = min(finite_y)
    y_max = max(finite_y)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    W, H = 1800, 1100
    left, right = 90, 1710
    top, bottom = 95, 1025
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    if font is None or small_font is None:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    x_max = max(1, len(x_values) - 1)
    x_lim = (0, x_max)
    y_lim = (y_min, y_max)
    draw_axis(draw, (left, top, right, bottom), x_lim[0], x_lim[1], y_lim[0], y_lim[1], title, "", small_font)
    draw_series(draw, (left, top, right, bottom), x_values, y_values, x_lim, y_lim, color, width=26)
    img.save(out_png)


def summarize(values):
    if not values:
        return "n/a"
    finite = [v for v in values if v is not None]
    if not finite:
        return "n/a"
    return f"{finite[0]:.3f} → {finite[-1]:.3f}"


def plot(data, out_png: Path, out_csv: Path):
    steps = data["steps"]
    loss = data["losses"]
    acc = data["accuracy"]
    invalid = data["invalid"]
    total = data["invalid_total"]
    times = data["step_times"]
    invalid_ratio = []
    for i in range(len(invalid)):
        if invalid[i] is None or total[i] is None or total[i] == 0:
            invalid_ratio.append(0.0)
        else:
            invalid_ratio.append(float(invalid[i]) / float(total[i]))

    # compact chart with 2 rows:
    # top: objective/quality trajectories
    # bottom: step-time and invalid-plan rate
    W, H = 4200, 2800
    left = 120
    right = 4080
    row1_top = 160
    row1_bottom = 1350
    row2_top = 1490
    row2_bottom = 2670
    margin_bottom = 2720

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 54)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    cfg = data["cfg"]
    title = (
        f"W&B Full-Run Training Trace • config: B={cfg['B']} G={cfg['G']} "
        f"reward={cfg['reward_mode']} beta={cfg['beta']}"
    )
    draw.text((left, 30), title, fill="#111111", font=font)

    # Top-left: loss
    loss_bbox = (left, row1_top, right - 220, row1_bottom)
    lvals = [v for v in loss if v is not None]
    x = list(range(len(loss)))
    draw_axis(
        draw,
        loss_bbox,
        0,
        max(1, len(loss) - 1),
        min(lvals) if lvals else 0,
        max(lvals) if lvals else 1,
        "Loss",
        "loss",
        small_font,
    )
    draw_series(draw, loss_bbox, x, loss, (0, max(1, len(loss) - 1)), (min(lvals) if lvals else 0, max(lvals) if lvals else 1), "#a43", width=3)
    final_loss = summarize(loss)
    draw.text((loss_bbox[2] - 140, loss_bbox[1] + 6), f"range {final_loss}", fill="#a43", font=small_font)

    # Top-right: outcome accuracy
    acc_bbox = (right - 180, row1_top, right, row1_bottom)
    avals = [v for v in acc if v is not None]
    if not avals:
        avals = [0.0]
    draw_axis(
        draw,
        acc_bbox,
        0,
        max(1, len(acc) - 1),
        0.0,
        1.0,
        "Outcome accuracy",
        "acc",
        small_font,
    )
    draw_series(draw, acc_bbox, list(range(len(acc))), acc, (0, max(1, len(acc) - 1)), (0.0, 1.0), "#16a085", width=3)
    valid_acc = [v for v in acc if v is not None]
    draw.text((acc_bbox[2] - 180, acc_bbox[1] + 6), f"final {valid_acc[-1]*100:.2f}%" if valid_acc else "final n/a", fill="#16a085", font=small_font)

    # Bottom: invalid-plan ratio only (remove step-time panel text for readability)
    ir_bbox = (left, row2_top, right, row2_bottom)
    ratios = invalid_ratio
    xvals = list(range(len(ratios)))
    draw_axis(
        draw,
        ir_bbox,
        0,
        max(1, len(ratios) - 1),
        0.0,
        1.0,
        "Invalid plans",
        "ratio",
        small_font,
    )
    draw_series(draw, ir_bbox, xvals, ratios, (0, max(1, len(ratios) - 1)), (0.0, 1.0), "#e67e22", width=3)

    img.save(out_png)

    # Export single-panel PNGs so each metric can be displayed as a separate readable chart.
    out_dir = out_png.parent
    render_panel(
        out_dir / "wandb_loss_chart.png",
        "Loss",
        loss,
        color="#a43",
        font=font,
        small_font=small_font,
    )
    render_panel(
        out_dir / "wandb_accuracy_chart.png",
        "Outcome Accuracy",
        acc,
        color="#16a085",
        font=font,
        small_font=small_font,
    )
    render_panel(
        out_dir / "wandb_invalid_ratio_chart.png",
        "Invalid Plan Ratio",
        invalid_ratio,
        color="#e67e22",
        font=font,
        small_font=small_font,
    )

    # CSV for reproducibility and future edits
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("step,loss,outcome_acc,invalid_ratio,invalid_num,invalid_denom,step_time_sec\n")
        for i in range(len(loss)):
            denom = total[i] if i < len(total) else 1
            f.write(
                f"{i},{loss[i] if loss[i] is not None else ''},{acc[i] if acc[i] is not None else ''},"
                f"{invalid_ratio[i] if i < len(invalid_ratio) else ''},"
                f"{invalid[i] if i < len(invalid) else ''},{denom},{times[i] if i < len(times) else ''}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="router_solver/wandb/latest-run/files/output.log", help="WandB output.log path")
    parser.add_argument("--png", default="router_solver/docs/assets/wandb_training_results_chart.png")
    parser.add_argument("--csv", default="router_solver/docs/assets/wandb_training_results_data.csv")
    args = parser.parse_args()

    data = parse_log(Path(args.log))
    out_png = Path(args.png)
    out_csv = Path(args.csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plot(data, out_png, out_csv)


if __name__ == "__main__":
    main()
