#!/usr/bin/env python3
"""
Parse one or more bench run directories and print a side-by-side comparison.

Usage
-----
    python scripts/bench/summarize.py outputs/bench/baseline__... outputs/bench/remote__...

Optional flags:
    --json            also emit a single JSON blob with the parsed metrics
    --csv FILE        also write the wide comparison table as CSV

Each run directory is expected to contain:
    walltime.csv      phase markers written by run_bench.sh
    gpu_metrics.csv   nvidia-smi poll output written by monitor_gpus.sh
    config.env        env-var snapshot
    trainer.log       full stdout of the training wrapper

The summary computes:
    - phase walltimes (script duration, trainer duration)
    - per-GPU averages and peaks (utilization %, mem MiB, power W)
    - aggregate "GPU-seconds" used (sum of util*dt) for cost normalization
    - first/last training step timestamps from trainer.log progress lines
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GpuStats:
    samples: int = 0
    util_avg: float = 0.0
    util_peak: float = 0.0
    mem_used_avg_mib: float = 0.0
    mem_used_peak_mib: float = 0.0
    mem_total_mib: float = 0.0
    power_avg_w: float = 0.0
    power_peak_w: float = 0.0
    energy_wh: float = 0.0  # power averaged over wall time, integrated
    gpu_seconds_busy: float = 0.0  # sum(util_pct/100 * dt) — dimensionless * seconds


@dataclass
class RunSummary:
    run_dir: Path
    config: dict[str, str] = field(default_factory=dict)
    walltime: dict[str, dict[str, Any]] = field(default_factory=dict)
    duration_s: dict[str, float] = field(default_factory=dict)
    per_gpu: dict[int, GpuStats] = field(default_factory=dict)
    aggregate: GpuStats = field(default_factory=GpuStats)
    trainer_log_summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- I/O
def parse_config_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def parse_walltime(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    if not path.is_file():
        return {}, {}
    rows: list[dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    phases: dict[str, dict[str, Any]] = {}
    for row in rows:
        phases[row["phase"]] = {
            "epoch_s": float(row["epoch_s"]),
            "iso_ts": row["iso_ts"],
        }
    durations: dict[str, float] = {}
    if "script_start" in phases and "script_end" in phases:
        durations["script_total_s"] = phases["script_end"]["epoch_s"] - phases["script_start"]["epoch_s"]
    if "trainer_launch" in phases and "trainer_exit" in phases:
        durations["trainer_s"] = phases["trainer_exit"]["epoch_s"] - phases["trainer_launch"]["epoch_s"]
    if "script_start" in phases and "trainer_launch" in phases:
        durations["pre_trainer_s"] = phases["trainer_launch"]["epoch_s"] - phases["script_start"]["epoch_s"]
    return phases, durations


def parse_gpu_metrics(path: Path) -> tuple[dict[int, GpuStats], GpuStats]:
    if not path.is_file():
        return {}, GpuStats()
    rows: list[dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}, GpuStats()

    by_gpu: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row.get("index") in (None, "", "NA"):
            continue
        try:
            idx = int(row["index"])
        except ValueError:
            continue

        def _f(key: str) -> float:
            val = row.get(key)
            if val is None or val == "" or val == "NA":
                return float("nan")
            try:
                return float(val)
            except ValueError:
                return float("nan")

        by_gpu[idx].append({
            "epoch_s": _f("epoch_s"),
            "util_gpu_pct": _f("util_gpu_pct"),
            "util_mem_pct": _f("util_mem_pct"),
            "mem_used_mib": _f("mem_used_mib"),
            "mem_total_mib": _f("mem_total_mib"),
            "power_w": _f("power_w"),
        })

    per_gpu: dict[int, GpuStats] = {}
    for idx, samples in by_gpu.items():
        stats = GpuStats(samples=len(samples))

        def _avg(key: str) -> float:
            vals = [s[key] for s in samples if not _isnan(s[key])]
            return sum(vals) / len(vals) if vals else float("nan")

        def _peak(key: str) -> float:
            vals = [s[key] for s in samples if not _isnan(s[key])]
            return max(vals) if vals else float("nan")

        stats.util_avg = _avg("util_gpu_pct")
        stats.util_peak = _peak("util_gpu_pct")
        stats.mem_used_avg_mib = _avg("mem_used_mib")
        stats.mem_used_peak_mib = _peak("mem_used_mib")
        stats.mem_total_mib = _peak("mem_total_mib")
        stats.power_avg_w = _avg("power_w")
        stats.power_peak_w = _peak("power_w")

        # Time-integrate util * dt (gpu-seconds busy) and power * dt (energy Wh).
        gs = 0.0
        wh = 0.0
        prev_t: float | None = None
        for s in samples:
            t = s["epoch_s"]
            if prev_t is not None:
                dt = t - prev_t
                if dt > 0:
                    if not _isnan(s["util_gpu_pct"]):
                        gs += (s["util_gpu_pct"] / 100.0) * dt
                    if not _isnan(s["power_w"]):
                        wh += (s["power_w"] * dt) / 3600.0
            prev_t = t
        stats.gpu_seconds_busy = gs
        stats.energy_wh = wh

        per_gpu[idx] = stats

    # Aggregate across GPUs.
    agg = GpuStats()
    if per_gpu:
        agg.samples = sum(s.samples for s in per_gpu.values())
        # Weighted by per-GPU sample count (assumes synchronized polling).
        agg.util_avg = _safe_mean([s.util_avg for s in per_gpu.values()])
        agg.util_peak = max((s.util_peak for s in per_gpu.values()), default=float("nan"))
        agg.mem_used_avg_mib = sum(s.mem_used_avg_mib for s in per_gpu.values() if not _isnan(s.mem_used_avg_mib))
        agg.mem_used_peak_mib = sum(s.mem_used_peak_mib for s in per_gpu.values() if not _isnan(s.mem_used_peak_mib))
        agg.mem_total_mib = sum(s.mem_total_mib for s in per_gpu.values() if not _isnan(s.mem_total_mib))
        agg.power_avg_w = sum(s.power_avg_w for s in per_gpu.values() if not _isnan(s.power_avg_w))
        agg.power_peak_w = sum(s.power_peak_w for s in per_gpu.values() if not _isnan(s.power_peak_w))
        agg.energy_wh = sum(s.energy_wh for s in per_gpu.values())
        agg.gpu_seconds_busy = sum(s.gpu_seconds_busy for s in per_gpu.values())

    return per_gpu, agg


_TQDM_RE = re.compile(
    r"\b(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*\[(?P<elapsed>[\d:]+)<.*?,\s*(?P<rate>[\d\.]+)\s*(?P<unit>s/it|it/s)"
)
# Emitted by run_logit_fusion_gspo_remote.sh once the teacher HTTP server
# answers /health. Lets us split "trainer time" from "teacher boot time".
_TEACHER_READY_RE = re.compile(r"teacher ready after\s+(?P<sec>\d+(?:\.\d+)?)s")


def parse_trainer_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    info: dict[str, Any] = {}
    last_match: dict[str, str] | None = None
    first_match: dict[str, str] | None = None
    with path.open(errors="replace") as f:
        for line in f:
            for m in _TQDM_RE.finditer(line):
                d = m.groupdict()
                if first_match is None:
                    first_match = d
                last_match = d
            tm = _TEACHER_READY_RE.search(line)
            if tm is not None:
                info["teacher_boot_s"] = float(tm.group("sec"))
    if last_match is not None:
        info["last_step"] = int(last_match["step"])
        info["total_steps"] = int(last_match["total"])
        info["last_elapsed_str"] = last_match["elapsed"]
        info["last_rate"] = float(last_match["rate"])
        info["last_rate_unit"] = last_match["unit"]
        if last_match["unit"] == "s/it":
            info["sec_per_step"] = float(last_match["rate"])
        else:  # it/s
            r = float(last_match["rate"])
            info["sec_per_step"] = (1.0 / r) if r > 0 else float("nan")
    return info


# ---------------------------------------------------------------------- helpers
def _isnan(x: float) -> bool:
    return x != x


def _safe_mean(values: list[float]) -> float:
    cleaned = [v for v in values if not _isnan(v)]
    return sum(cleaned) / len(cleaned) if cleaned else float("nan")


def _fmt(val: Any, spec: str = "") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if val != val:
            return "—"
        if spec:
            return format(val, spec)
        return f"{val:.2f}"
    return str(val)


def summarize_run(run_dir: Path) -> RunSummary:
    s = RunSummary(run_dir=run_dir)
    s.config = parse_config_env(run_dir / "config.env")
    s.walltime, s.duration_s = parse_walltime(run_dir / "walltime.csv")
    s.per_gpu, s.aggregate = parse_gpu_metrics(run_dir / "gpu_metrics.csv")
    s.trainer_log_summary = parse_trainer_log(run_dir / "trainer.log")
    return s


# ---------------------------------------------------------------------- output
def render_table(rows: list[list[str]], align: list[str] | None = None) -> str:
    if not rows:
        return ""
    cols = len(rows[0])
    widths = [max(len(r[c]) for r in rows) for c in range(cols)]
    align = align or ["l"] * cols
    out_lines: list[str] = []
    for i, r in enumerate(rows):
        cells = []
        for c, val in enumerate(r):
            if align[c] == "r":
                cells.append(val.rjust(widths[c]))
            else:
                cells.append(val.ljust(widths[c]))
        out_lines.append("  ".join(cells))
        if i == 0:
            out_lines.append("  ".join("-" * w for w in widths))
    return "\n".join(out_lines)


def render_summary(summaries: list[RunSummary]) -> str:
    out: list[str] = []
    headers = ["metric"] + [s.run_dir.name for s in summaries]
    align = ["l"] + ["r"] * len(summaries)

    def row(label: str, fn) -> list[str]:
        return [label] + [_fmt(fn(s)) for s in summaries]

    # ---- run config snapshot ----
    out.append("== Run config ==")
    cfg_keys = [
        "MODE",
        "PROFILE",
        "MODEL_NAME_OR_PATH",
        "TEACHER_MODEL_NAME_OR_PATH",
        "STUDENT_GPUS",
        "TEACHER_GPUS",
        "BASELINE_GPUS",
        "BASELINE_STRATEGY",
        "BASELINE_DDP_TOKEN_MODE",
        "BASELINE_DDP_REFERENCE_PER_DEVICE_TRAIN_BATCH_SIZE",
        "BASELINE_DDP_REFERENCE_TOKENS_PER_STEP",
        "NUM_PROCESSES",
        "PER_DEVICE_TRAIN_BATCH_SIZE",
        "NUM_GENERATIONS",
        "STEPS_PER_GENERATION",
        "GRADIENT_ACCUMULATION_STEPS",
        "MAX_PROMPT_LENGTH",
        "MAX_COMPLETION_LENGTH",
        "MAX_STEPS",
    ]
    rows = [headers]
    for k in cfg_keys:
        rows.append([k] + [s.config.get(k, "—") for s in summaries])
    out.append(render_table(rows, align))
    out.append("")

    # ---- walltime ----
    def _trainer_pure(s: RunSummary) -> float | None:
        t = s.duration_s.get("trainer_s")
        boot = s.trainer_log_summary.get("teacher_boot_s")
        if t is None:
            return None
        return t - boot if isinstance(boot, (int, float)) else t

    out.append("== Walltime ==")
    rows = [headers]
    rows.append(row("script_total_s",  lambda s: s.duration_s.get("script_total_s")))
    rows.append(row("pre_trainer_s",   lambda s: s.duration_s.get("pre_trainer_s")))
    rows.append(row("trainer_s",       lambda s: s.duration_s.get("trainer_s")))
    rows.append(row("teacher_boot_s",  lambda s: s.trainer_log_summary.get("teacher_boot_s")))
    rows.append(row("trainer_pure_s",  _trainer_pure))
    rows.append(row("trainer_steps",   lambda s: s.trainer_log_summary.get("last_step")))
    rows.append(row("sec_per_step",    lambda s: s.trainer_log_summary.get("sec_per_step")))
    out.append(render_table(rows, align))
    out.append("")

    # ---- aggregate GPU stats ----
    out.append("== Aggregate GPU (sum across all monitored GPUs) ==")
    rows = [headers]
    rows.append(row("util_avg_pct (mean of GPUs)", lambda s: s.aggregate.util_avg))
    rows.append(row("util_peak_pct (any GPU)",     lambda s: s.aggregate.util_peak))
    rows.append(row("mem_used_avg_GiB",            lambda s: s.aggregate.mem_used_avg_mib / 1024.0))
    rows.append(row("mem_used_peak_GiB",           lambda s: s.aggregate.mem_used_peak_mib / 1024.0))
    rows.append(row("power_avg_W",                 lambda s: s.aggregate.power_avg_w))
    rows.append(row("power_peak_W",                lambda s: s.aggregate.power_peak_w))
    rows.append(row("energy_Wh",                   lambda s: s.aggregate.energy_wh))
    rows.append(row("gpu_seconds_busy",            lambda s: s.aggregate.gpu_seconds_busy))
    out.append(render_table(rows, align))
    out.append("")

    # ---- per-GPU breakdown ----
    out.append("== Per-GPU detail ==")
    all_indices = sorted({i for s in summaries for i in s.per_gpu.keys()})
    rows = [["gpu_idx", "metric"] + [s.run_dir.name for s in summaries]]
    align2 = ["r", "l"] + ["r"] * len(summaries)
    for idx in all_indices:
        for label, attr, scale in [
            ("util_avg_%",     "util_avg",         1.0),
            ("util_peak_%",    "util_peak",        1.0),
            ("mem_avg_GiB",    "mem_used_avg_mib", 1.0 / 1024.0),
            ("mem_peak_GiB",   "mem_used_peak_mib",1.0 / 1024.0),
            ("power_avg_W",    "power_avg_w",      1.0),
            ("gpu_sec_busy",   "gpu_seconds_busy", 1.0),
        ]:
            cells = [str(idx), label]
            for s in summaries:
                stats = s.per_gpu.get(idx)
                if stats is None:
                    cells.append("—")
                else:
                    cells.append(_fmt(getattr(stats, attr) * scale))
            rows.append(cells)
    out.append(render_table(rows, align2))
    out.append("")

    return "\n".join(out)


def runs_to_json(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    blob: list[dict[str, Any]] = []
    for s in summaries:
        blob.append({
            "run_dir": str(s.run_dir),
            "config": s.config,
            "walltime_phases": s.walltime,
            "duration_s": s.duration_s,
            "trainer_log": s.trainer_log_summary,
            "aggregate_gpu": s.aggregate.__dict__,
            "per_gpu": {idx: stats.__dict__ for idx, stats in s.per_gpu.items()},
        })
    return blob


_CONFIG_COLS: list[str] = [
    "MODE",
    "PROFILE",
    "MODEL_NAME_OR_PATH",
    "TEACHER_MODEL_NAME_OR_PATH",
    "STUDENT_GPUS",
    "TEACHER_GPUS",
    "BASELINE_GPUS",
    "BASELINE_STRATEGY",
    "BASELINE_DDP_TOKEN_MODE",
    "BASELINE_DDP_REFERENCE_PER_DEVICE_TRAIN_BATCH_SIZE",
    "BASELINE_DDP_REFERENCE_TOKENS_PER_STEP",
    "NUM_PROCESSES",
    "PER_DEVICE_TRAIN_BATCH_SIZE",
    "NUM_GENERATIONS",
    "STEPS_PER_GENERATION",
    "GRADIENT_ACCUMULATION_STEPS",
    "MAX_PROMPT_LENGTH",
    "MAX_COMPLETION_LENGTH",
    "MAX_STEPS",
]


def _row_for_run(s: RunSummary, gpu_indices: list[int]) -> dict[str, Any]:
    """Flatten a RunSummary into one wide CSV row."""
    boot = s.trainer_log_summary.get("teacher_boot_s")
    trainer_s = s.duration_s.get("trainer_s")
    trainer_pure = (
        trainer_s - boot
        if isinstance(trainer_s, (int, float)) and isinstance(boot, (int, float))
        else trainer_s
    )

    # Derived per-step token throughput, useful for normalizing across profiles.
    bs = _safe_int(s.config.get("PER_DEVICE_TRAIN_BATCH_SIZE"))
    gen = _safe_int(s.config.get("NUM_GENERATIONS"))
    comp = _safe_int(s.config.get("MAX_COMPLETION_LENGTH"))
    np_ = _safe_int(s.config.get("NUM_PROCESSES"))
    tokens_per_step: float | str = ""
    if bs and gen and comp and np_:
        tokens_per_step = bs * gen * comp * np_

    sec_per_step = s.trainer_log_summary.get("sec_per_step")
    tokens_per_sec: float | str = ""
    if isinstance(tokens_per_step, (int, float)) and isinstance(sec_per_step, (int, float)) and sec_per_step > 0:
        tokens_per_sec = tokens_per_step / sec_per_step

    row: dict[str, Any] = {
        "run_dir": str(s.run_dir),
        "run_name": s.run_dir.name,
    }
    for k in _CONFIG_COLS:
        row[f"cfg_{k}"] = s.config.get(k, "")

    row.update({
        "wall_script_total_s": s.duration_s.get("script_total_s", ""),
        "wall_pre_trainer_s": s.duration_s.get("pre_trainer_s", ""),
        "wall_trainer_s": trainer_s if trainer_s is not None else "",
        "wall_teacher_boot_s": boot if boot is not None else "",
        "wall_trainer_pure_s": trainer_pure if trainer_pure is not None else "",
        "trainer_steps": s.trainer_log_summary.get("last_step", ""),
        "sec_per_step": sec_per_step if sec_per_step is not None else "",
        "tokens_generated_per_step": tokens_per_step,
        "tokens_generated_per_sec": tokens_per_sec,
        "agg_util_avg_pct": s.aggregate.util_avg,
        "agg_util_peak_pct": s.aggregate.util_peak,
        "agg_mem_used_avg_gib": s.aggregate.mem_used_avg_mib / 1024.0,
        "agg_mem_used_peak_gib": s.aggregate.mem_used_peak_mib / 1024.0,
        "agg_power_avg_w": s.aggregate.power_avg_w,
        "agg_power_peak_w": s.aggregate.power_peak_w,
        "agg_energy_wh": s.aggregate.energy_wh,
        "agg_gpu_seconds_busy": s.aggregate.gpu_seconds_busy,
    })

    for idx in gpu_indices:
        st = s.per_gpu.get(idx)
        prefix = f"gpu{idx}_"
        if st is None:
            row[prefix + "util_avg_pct"] = ""
            row[prefix + "util_peak_pct"] = ""
            row[prefix + "mem_used_avg_gib"] = ""
            row[prefix + "mem_used_peak_gib"] = ""
            row[prefix + "power_avg_w"] = ""
            row[prefix + "gpu_seconds_busy"] = ""
        else:
            row[prefix + "util_avg_pct"] = st.util_avg
            row[prefix + "util_peak_pct"] = st.util_peak
            row[prefix + "mem_used_avg_gib"] = st.mem_used_avg_mib / 1024.0
            row[prefix + "mem_used_peak_gib"] = st.mem_used_peak_mib / 1024.0
            row[prefix + "power_avg_w"] = st.power_avg_w
            row[prefix + "gpu_seconds_busy"] = st.gpu_seconds_busy
    return row


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def runs_to_csv(summaries: list[RunSummary], path: Path) -> None:
    """Wide CSV: one row per run, columns for config + walltime + per-GPU."""
    gpu_indices = sorted({idx for s in summaries for idx in s.per_gpu.keys()})
    rows = [_row_for_run(s, gpu_indices) for s in summaries]
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------- main
def _default_csv_path(summaries: list[RunSummary]) -> Path:
    """Place summary.csv next to the run dirs (their common parent)."""
    parents = {s.run_dir.parent for s in summaries if s.run_dir.parent.exists()}
    parent = next(iter(parents)) if len(parents) == 1 else Path("outputs/bench")
    ts = __import__("time").strftime("%Y%m%d-%H%M%S")
    return parent / f"summary__{ts}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Bench run directories produced by run_bench.sh")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON instead of a table.")
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Write a wide comparison CSV here. Defaults to "
             "<common-parent>/summary__<timestamp>.csv unless --no-csv is given.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip writing the comparison CSV.")
    args = parser.parse_args(argv)

    summaries = [summarize_run(Path(d)) for d in args.run_dirs]

    if args.json:
        json.dump(runs_to_json(summaries), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_summary(summaries))

    if not args.no_csv:
        csv_path = args.csv if args.csv is not None else _default_csv_path(summaries)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        runs_to_csv(summaries, csv_path)
        sys.stderr.write(f"\nwrote CSV: {csv_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
