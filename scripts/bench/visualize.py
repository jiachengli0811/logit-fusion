#!/usr/bin/env python3
"""
Create a standalone HTML visualization from one or more bench summary CSVs.

Examples
--------
    python scripts/bench/visualize.py outputs/bench/summary__20260503-164642.csv
    python scripts/bench/visualize.py --latest
    python scripts/bench/visualize.py outputs/bench/summary__*.csv --out outputs/bench/report.html

The input CSV is the wide file produced by scripts/bench/summarize.py. The
output is dependency-free HTML with inline SVG charts, so it can be opened in a
browser or attached to a report without needing pandas, matplotlib, or a server.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


LABEL_ORDER = ["baseline", "parallel", "baseline-ddp-global", "baseline-ddp-rank", "baseline-ddp"]
PROFILE_ORDER = ["tiny", "stress", "xl"]
COLORS = {
    "baseline": "#60a5fa",
    "parallel": "#059669",
    "baseline-ddp-global": "#d97706",
    "baseline-ddp-rank": "#b45309",
    "baseline-ddp": "#d97706",
    "other": "#64748b",
}


@dataclass
class RunRow:
    source_csv: Path
    raw: dict[str, str]

    @property
    def run_name(self) -> str:
        return self.raw.get("run_name") or Path(self.raw.get("run_dir", "")).name

    @property
    def profile(self) -> str:
        value = self.raw.get("cfg_PROFILE", "").strip()
        if value:
            return value
        comp = self.raw.get("cfg_MAX_COMPLETION_LENGTH", "").strip()
        if comp == "256":
            return "tiny"
        if comp == "1024":
            return "stress"
        if comp == "2048":
            return "xl"
        return "unknown"

    @property
    def label(self) -> str:
        mode = self.raw.get("cfg_MODE", "").strip()
        strategy = self.raw.get("cfg_BASELINE_STRATEGY", "").strip()
        token_mode = self.raw.get("cfg_BASELINE_DDP_TOKEN_MODE", "").strip()
        name = self.run_name
        if mode == "remote" or name.startswith("remote") or name.startswith("parallel"):
            return "parallel"
        if strategy == "ddp" or name.startswith("baseline-ddp"):
            if token_mode == "same_global" or name.startswith("baseline-ddp-global"):
                return "baseline-ddp-global"
            if token_mode == "same_per_rank" or name.startswith("baseline-ddp-rank"):
                return "baseline-ddp-rank"
            return "baseline-ddp"
        if mode == "baseline" or name.startswith("baseline"):
            return "baseline"
        return "other"


@dataclass
class GroupRow:
    profile: str
    label: str
    runs: list[RunRow] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def read_rows(paths: Iterable[Path]) -> list[RunRow]:
    rows: list[RunRow] = []
    for path in paths:
        with path.open(newline="") as f:
            for raw in csv.DictReader(f):
                rows.append(RunRow(source_csv=path, raw=raw))
    return rows


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def mean(values: Iterable[float]) -> float:
    cleaned = [v for v in values if math.isfinite(v)]
    return statistics.fmean(cleaned) if cleaned else float("nan")


def summarize_groups(rows: list[RunRow]) -> list[GroupRow]:
    grouped: dict[tuple[str, str], list[RunRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.profile, row.label)].append(row)

    out: list[GroupRow] = []
    for (profile, label), runs in grouped.items():
        metrics: dict[str, float] = {}
        for key in [
            "trainer_steps",
            "sec_per_step",
            "tokens_generated_per_step",
            "tokens_generated_per_sec",
            "wall_trainer_s",
            "wall_teacher_boot_s",
            "wall_trainer_pure_s",
            "agg_util_avg_pct",
            "agg_mem_used_peak_gib",
            "agg_power_avg_w",
            "agg_energy_wh",
            "agg_gpu_seconds_busy",
        ]:
            metrics[key] = mean(to_float(r.raw.get(key)) for r in runs)

        total_tokens = metrics["trainer_steps"] * metrics["tokens_generated_per_step"]
        metrics["total_generated_tokens"] = total_tokens
        metrics["tokens_per_gpu_busy_s"] = (
            total_tokens / metrics["agg_gpu_seconds_busy"]
            if total_tokens > 0 and metrics["agg_gpu_seconds_busy"] > 0
            else float("nan")
        )
        metrics["energy_wh_per_mtoken"] = (
            metrics["agg_energy_wh"] / (total_tokens / 1_000_000)
            if total_tokens > 0 and metrics["agg_energy_wh"] > 0
            else float("nan")
        )
        out.append(GroupRow(profile=profile, label=label, runs=runs, metrics=metrics))

    return sorted(out, key=lambda g: (profile_rank(g.profile), label_rank(g.label), g.label))


def profile_rank(profile: str) -> tuple[int, str]:
    try:
        return (PROFILE_ORDER.index(profile), profile)
    except ValueError:
        return (len(PROFILE_ORDER), profile)


def label_rank(label: str) -> int:
    try:
        return LABEL_ORDER.index(label)
    except ValueError:
        return len(LABEL_ORDER)


def fmt(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_int(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:,.0f}"


def pct(delta: float) -> str:
    if not math.isfinite(delta):
        return "-"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def lower_better_delta(new: float, baseline: float) -> str:
    if not math.isfinite(new) or not math.isfinite(baseline) or baseline == 0:
        return "-"
    if new <= baseline:
        return f"{(1 - new / baseline) * 100:.1f}% lower"
    return f"{(new / baseline - 1) * 100:.1f}% higher"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def comparison_rows(groups: list[GroupRow]) -> list[dict[str, str]]:
    by_key = {(g.profile, g.label): g for g in groups}
    profiles = sorted({g.profile for g in groups}, key=profile_rank)
    out: list[dict[str, str]] = []
    for profile in profiles:
        base = by_key.get((profile, "baseline"))
        parallel = by_key.get((profile, "parallel"))
        ddp_groups = [
            by_key[(profile, label)]
            for label in LABEL_ORDER
            if label.startswith("baseline-ddp") and (profile, label) in by_key
        ]
        if base and parallel:
            b = base.metrics
            r = parallel.metrics
            out.append({
                "profile": profile,
                "comparison": "parallel vs baseline",
                "step_time": f"{fmt(r['sec_per_step'], 2)} vs {fmt(b['sec_per_step'], 2)} s ({pct((b['sec_per_step'] / r['sec_per_step'] - 1) * 100)})",
                "throughput": f"{fmt(r['tokens_generated_per_sec'], 1)} vs {fmt(b['tokens_generated_per_sec'], 1)} tok/s",
                "memory": f"{fmt(r['agg_mem_used_peak_gib'], 1)} vs {fmt(b['agg_mem_used_peak_gib'], 1)} GiB ({lower_better_delta(r['agg_mem_used_peak_gib'], b['agg_mem_used_peak_gib'])})",
                "energy": f"{fmt(r['agg_energy_wh'], 1)} vs {fmt(b['agg_energy_wh'], 1)} Wh ({lower_better_delta(r['agg_energy_wh'], b['agg_energy_wh'])})",
            })
        for ddp in ddp_groups:
            d = ddp.metrics
            out.append({
                "profile": profile,
                "comparison": ddp.label,
                "step_time": f"{fmt(d['sec_per_step'], 2)} s",
                "throughput": f"{fmt(d['tokens_generated_per_sec'], 1)} tok/s",
                "memory": f"{fmt(d['agg_mem_used_peak_gib'], 1)} GiB",
                "energy": f"{fmt(d['agg_energy_wh'], 1)} Wh",
            })
    return out


def chart_svg(
    groups: list[GroupRow],
    key: str,
    title: str,
    unit: str,
    digits: int = 1,
    *,
    show_title: bool = True,
    show_profile: bool = True,
    left: int | None = 210,
    label_right_align: bool = False,
    label_pad: int = 6,
) -> str:
    rows = [(g, g.metrics.get(key, float("nan"))) for g in groups if math.isfinite(g.metrics.get(key, float("nan")))]
    if not rows:
        return ""

    width = 980
    right = 110
    top = 42 if show_title else 18
    row_h = 34
    gap = 8
    height = top + len(rows) * row_h + 24
    max_value = max(value for _, value in rows)
    bar_w = width - left - right

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{esc(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
    ]
    if show_title:
        parts.append(f'<text x="0" y="20" font-size="16" font-weight="700" fill="#0f172a">{esc(title)}</text>')

    label_texts: list[str] = []
    for group, _ in rows:
        label_texts.append(f"{group.profile} / {group.label}" if show_profile else group.label)

    if left is None:
        approx_px = max(len(text) for text in label_texts) * 6 + 14
        left = int(min(180, max(70, approx_px)))

    label_x = (left - label_pad) if label_right_align else 0
    label_anchor = "end" if label_right_align else "start"
    for idx, ((group, value), label_text) in enumerate(zip(rows, label_texts, strict=False)):
        y = top + idx * row_h
        color = COLORS.get(group.label, COLORS["other"])
        w = 0 if max_value <= 0 else (value / max_value) * bar_w
        value_text = f"{fmt(value, digits)} {unit}".strip()
        parts.extend([
            f'<text x="{label_x}" y="{y + 18}" text-anchor="{label_anchor}" font-size="12" fill="#334155">{esc(label_text)}</text>',
            f'<rect x="{left}" y="{y}" width="{bar_w}" height="{row_h - gap}" rx="3" fill="#eef2f7" />',
            f'<rect class="bar" x="{left}" y="{y}" width="{w:.2f}" height="{row_h - gap}" rx="3" fill="{color}" />',
            f'<text x="{left + w + 8 if w < bar_w - 72 else left + w - 8}" y="{y + 18}" text-anchor="{"start" if w < bar_w - 72 else "end"}" font-size="12" font-weight="700" fill="#0f172a">{esc(value_text)}</text>',
        ])
    parts.append("</svg>")
    return "\n".join(parts)


def group_table(groups: list[GroupRow]) -> str:
    headers = [
        "profile",
        "run type",
        "runs",
        "sec/step",
        "gen tok/s",
        "tokens/step",
        "peak mem GiB",
        "GPU busy s",
        "energy Wh",
        "tok/GPU-busy-s",
        "Wh/Mtok",
    ]
    body = []
    for g in groups:
        m = g.metrics
        body.append([
            g.profile,
            g.label,
            str(len(g.runs)),
            fmt(m["sec_per_step"], 2),
            fmt(m["tokens_generated_per_sec"], 1),
            fmt_int(m["tokens_generated_per_step"]),
            fmt(m["agg_mem_used_peak_gib"], 1),
            fmt(m["agg_gpu_seconds_busy"], 1),
            fmt(m["agg_energy_wh"], 1),
            fmt(m["tokens_per_gpu_busy_s"], 1),
            fmt(m["energy_wh_per_mtoken"], 1),
        ])
    return html_table(headers, body)


def comparisons_table(groups: list[GroupRow]) -> str:
    rows = comparison_rows(groups)
    headers = ["profile", "comparison", "step time", "throughput", "peak memory", "energy"]
    body = [
        [r["profile"], r["comparison"], r["step_time"], r["throughput"], r["memory"], r["energy"]]
        for r in rows
    ]
    return html_table(headers, body)


def raw_table(rows: list[RunRow]) -> str:
    headers = [
        "run",
        "profile",
        "type",
        "processes",
        "batch",
        "gens",
        "max comp",
        "sec/step",
        "tok/s",
        "peak mem GiB",
        "energy Wh",
    ]
    body = []
    for r in sorted(rows, key=lambda row: (profile_rank(row.profile), label_rank(row.label), row.run_name)):
        body.append([
            r.run_name,
            r.profile,
            r.label,
            r.raw.get("cfg_NUM_PROCESSES", ""),
            r.raw.get("cfg_PER_DEVICE_TRAIN_BATCH_SIZE", ""),
            r.raw.get("cfg_NUM_GENERATIONS", ""),
            r.raw.get("cfg_MAX_COMPLETION_LENGTH", ""),
            fmt(to_float(r.raw.get("sec_per_step")), 2),
            fmt(to_float(r.raw.get("tokens_generated_per_sec")), 1),
            fmt(to_float(r.raw.get("agg_mem_used_peak_gib")), 1),
            fmt(to_float(r.raw.get("agg_energy_wh")), 1),
        ])
    return html_table(headers, body)


def html_table(headers: list[str], body: list[list[str]]) -> str:
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{esc(h)}</th>" for h in headers)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend(f"<td>{esc(cell)}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_html(rows: list[RunRow], groups: list[GroupRow], sources: list[Path]) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    source_text = ", ".join(str(p) for p in sources)
    chart_blocks = [
        chart_svg(
            groups,
            key,
            title,
            unit,
            digits,
            show_title=True,
            show_profile=True,
            left=210,
            label_right_align=False,
        )
        for key, title, unit, digits in chart_specs()
    ]
    chart_panels = "\n".join(f'<div class="panel">{chart}</div>' for chart in chart_blocks if chart)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Logit Fusion Bench Comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #dbe3ef;
      --soft: #eef2f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 42px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; font-weight: 750; }}
    h2 {{ margin: 30px 0 12px; font-size: 18px; }}
    .meta {{ color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }}
    .note {{
      margin-top: 18px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      color: #334155;
    }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 16px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; color: #334155; font-size: 13px; }}
    .swatch {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    .chart {{ width: 100%; min-width: 760px; display: block; }}
    .chart-title {{ font-size: 16px; font-weight: 700; fill: var(--text); }}
    .axis-label {{ font-size: 12px; fill: #334155; }}
    .bar-bg {{ fill: var(--soft); }}
    .bar-value {{ font-size: 12px; fill: #0f172a; font-weight: 650; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; }}
    th {{ background: #eef2f7; color: #334155; font-weight: 700; white-space: nowrap; }}
    td {{ white-space: nowrap; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Logit Fusion Bench Comparison</h1>
  <div class="meta">Generated {esc(generated)} from {esc(source_text)}</div>
  <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:{COLORS['baseline']}"></span>baseline</span>
        <span class="legend-item"><span class="swatch" style="background:{COLORS['parallel']}"></span>parallel</span>
    <span class="legend-item"><span class="swatch" style="background:{COLORS['baseline-ddp-global']}"></span>baseline-ddp-global</span>
    <span class="legend-item"><span class="swatch" style="background:{COLORS['baseline-ddp-rank']}"></span>baseline-ddp-rank</span>
    <span class="legend-item"><span class="swatch" style="background:{COLORS['baseline-ddp']}"></span>baseline-ddp (old)</span>
  </div>
  <div class="note">
    Repeated runs are averaged by <code>profile</code> and run type. Blank-strategy baseline rows are treated as
        <code>baseline</code>. Token-normalized DDP uses <code>baseline-ddp-global</code>; old same-per-rank DDP
    uses <code>baseline-ddp-rank</code> because it increases generated-token work per step when
    <code>NUM_PROCESSES &gt; 1</code>.
  </div>

  <h2>Headline Comparisons</h2>
  {comparisons_table(groups)}

  <h2>Grouped Metrics</h2>
  {group_table(groups)}

  <h2>Charts</h2>
  <div class="grid">
    {chart_panels}
  </div>

  <h2>Raw Runs</h2>
  {raw_table(rows)}
</main>
</body>
</html>
"""


def chart_specs() -> list[tuple[str, str, str, int]]:
    return [
        ("sec_per_step", "Step Time", "s", 2),
        ("tokens_generated_per_sec", "Generated Token Throughput", "tok/s", 1),
        ("agg_mem_used_peak_gib", "Peak GPU Memory", "GiB", 1),
        ("agg_gpu_seconds_busy", "GPU Busy Seconds", "s", 1),
        ("agg_energy_wh", "Energy", "Wh", 1),
        ("tokens_per_gpu_busy_s", "Generated Tokens Per GPU-Busy-Second", "tok/s", 1),
        ("energy_wh_per_mtoken", "Energy Per Million Generated Tokens", "Wh/Mtok", 1),
    ]


def combine_svgs_vertical(svgs: list[str], *, pad: int = 18) -> str:
    if not svgs:
        return ""

    parsed: list[tuple[float, float, ET.Element]] = []
    for svg_text in svgs:
        el = ET.fromstring(svg_text)
        view_box = el.attrib.get("viewBox", "0 0 0 0").split()
        if len(view_box) != 4:
            continue
        width = float(view_box[2])
        height = float(view_box[3])
        parsed.append((width, height, el))

    if not parsed:
        return ""

    max_width = max(w for w, _, _ in parsed)
    total_height = sum(h for _, h, _ in parsed) + pad * (len(parsed) - 1)
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {max_width:g} {total_height:g}",
            "width": f"{max_width:g}",
            "height": f"{total_height:g}",
        },
    )

    root.append(
        ET.Element(
            "rect",
            {
                "x": "0",
                "y": "0",
                "width": f"{max_width:g}",
                "height": f"{total_height:g}",
                "fill": "#ffffff",
            },
        )
    )

    y = 0.0
    for _, h, el in parsed:
        el.attrib["x"] = "0"
        el.attrib["y"] = f"{y:g}"
        root.append(el)
        y += h + pad

    return ET.tostring(root, encoding="unicode")


def export_charts(groups: list[GroupRow], export_dir: Path, *, write_png: bool) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    svg_blocks: list[tuple[str, str]] = []
    for key, title, unit, digits in chart_specs():
        svg = chart_svg(
            groups,
            key,
            title,
            unit,
            digits,
            show_title=False,
            show_profile=False,
            left=140,
            label_right_align=True,
            label_pad=4,
        )
        if svg:
            svg_blocks.append((key, svg))

    for key, svg in svg_blocks:
        (export_dir / f"{key}.svg").write_text(svg)

    combined_svg = combine_svgs_vertical([svg for _, svg in svg_blocks])
    if combined_svg:
        (export_dir / "combined.svg").write_text(combined_svg)

    if not write_png:
        return

    try:
        import cairosvg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PNG export requires cairosvg. Install with: pip install cairosvg"
        ) from exc

    for key, svg in svg_blocks:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(export_dir / f"{key}.png"))

    if combined_svg:
        cairosvg.svg2png(bytestring=combined_svg.encode("utf-8"), write_to=str(export_dir / "combined.png"))


def latest_summary() -> Path:
    candidates = sorted(Path("outputs/bench").glob("summary__*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("no outputs/bench/summary__*.csv files found")
    return candidates[-1]


def default_output(inputs: list[Path]) -> Path:
    if len(inputs) == 1:
        return inputs[0].with_suffix(".html")
    parent = inputs[0].parent if inputs else Path("outputs/bench")
    return parent / f"bench_visual__{time.strftime('%Y%m%d-%H%M%S')}.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary_csv", nargs="*", type=Path, help="summary__*.csv files from summarize.py")
    parser.add_argument("--latest", action="store_true", help="Use the newest outputs/bench/summary__*.csv")
    parser.add_argument("--out", type=Path, default=None, help="HTML output path")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="If set, export chart images into this directory (always writes SVG; add --png for PNG)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="With --export-dir, also render PNGs (requires cairosvg)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [p for p in args.summary_csv]
    if args.latest:
        inputs = [latest_summary()]
    if not inputs:
        inputs = [latest_summary()]

    missing = [p for p in inputs if not p.is_file()]
    if missing:
        sys.stderr.write("missing input CSV(s): " + ", ".join(str(p) for p in missing) + "\n")
        return 2

    rows = read_rows(inputs)
    if not rows:
        sys.stderr.write("no rows found in input CSV(s)\n")
        return 2

    groups = summarize_groups(rows)

    if args.export_dir is not None:
        try:
            export_charts(groups, args.export_dir, write_png=args.png)
        except RuntimeError as exc:
            sys.stderr.write(str(exc) + "\n")
            return 2

    out_path = args.out or default_output(inputs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(rows, groups, inputs))
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
