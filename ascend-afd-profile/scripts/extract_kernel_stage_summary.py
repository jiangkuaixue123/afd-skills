#!/usr/bin/env python3
"""Summarize per-microbatch stage durations from Ascend kernel_details.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def detect_marker(op_name: str) -> Optional[str]:
    normalized = normalize(op_name)
    if normalized.startswith("a2e"):
        return "A2e"
    if normalized.startswith("e2a"):
        return "E2a"
    return None


def infer_profile_side(csv_path: Path) -> str:
    normalized_parts = [normalize(part) for part in csv_path.parts]
    if "attention" in normalized_parts:
        return "attention"
    if "ffn" in normalized_parts:
        return "ffn"
    raise ValueError(
        "无法从路径推断侧别，请确保 kernel_details.csv 位于 profile/attention 或 profile/ffn 目录下。"
    )


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass
class KernelRow:
    index: int
    name: str
    start_us: float
    duration_us: float
    marker: Optional[str]


def load_rows(csv_path: Path) -> List[KernelRow]:
    rows: List[KernelRow] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            name = (row.get("Name") or row.get("OP Type") or row.get("Op Name") or "").strip()
            if not name:
                continue
            start_us = parse_float(row.get("Start Time(us)"))
            duration_us = parse_float(row.get("Duration(us)"))
            if start_us is None or duration_us is None:
                continue
            rows.append(
                KernelRow(
                    index=index,
                    name=name,
                    start_us=start_us,
                    duration_us=duration_us,
                    marker=detect_marker(name),
                )
            )
    rows.sort(key=lambda row: (row.start_us, row.index))
    return rows


def trim_extremes(values: List[float]) -> List[float]:
    if len(values) <= 2:
        return list(values)
    sorted_values = sorted(values)
    return sorted_values[1:-1]


def build_stage_samples(rows: List[KernelRow], side: str) -> Dict[str, List[float]]:
    stage_name = "attn_side_us" if side == "attention" else "ffn_side_us"
    stage_samples = {stage_name: []}

    marker_positions = [idx for idx, row in enumerate(rows) if row.marker]
    for marker_idx, next_marker_idx in zip(marker_positions, marker_positions[1:]):
        left = rows[marker_idx]
        right = rows[next_marker_idx]
        if side == "attention":
            is_target_window = left.marker == "E2a" and right.marker == "A2e"
        else:
            is_target_window = left.marker == "A2e" and right.marker == "E2a"
        if not is_target_window:
            continue

        total_duration = sum(
            row.duration_us
            for row in rows[marker_idx + 1 : next_marker_idx]
            if not row.marker
        )
        stage_samples[stage_name].append(total_duration)

    return stage_samples


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    trimmed = trim_extremes(values)
    if not trimmed:
        return {
            "raw_count": len(values),
            "trimmed_count": len(trimmed),
            "mean": None,
            "min": None,
            "max": None,
            "p75": None,
            "p90": None,
            "p99": None,
        }

    return {
        "raw_count": len(values),
        "trimmed_count": len(trimmed),
        "mean": sum(trimmed) / len(trimmed),
        "min": min(trimmed),
        "max": max(trimmed),
        "p75": percentile(trimmed, 0.75),
        "p90": percentile(trimmed, 0.90),
        "p99": percentile(trimmed, 0.99),
    }


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 kernel_details.csv 统计 AFD Attention 侧每个 microbatch 的阶段耗时。"
    )
    parser.add_argument("kernel_details_csv", help="kernel_details.csv 路径")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="输出格式，默认 markdown",
    )
    return parser.parse_args(argv)


def render_markdown(
    csv_path: Path,
    side: str,
    summaries: Dict[str, Dict[str, Optional[float]]],
) -> str:
    labels = {
        "attn_side_us": "Attention 侧 microbatch 执行时间（E2a -> 下一次 A2e 之间，不含 marker）",
        "ffn_side_us": "FFN 侧 microbatch 执行时间（A2e -> 下一次 E2a 之间，不含 marker）",
    }
    summary_key = "attn_side_us" if side == "attention" else "ffn_side_us"

    lines = [
        "## Kernel Stage Summary",
        "",
        f"- 输入文件: `{csv_path}`",
        f"- 侧别: `{side}`",
        "- 极值处理: 每组样本默认去掉 1 个最小值和 1 个最大值",
    ]
    stats = summaries[summary_key]
    lines.extend(["", f"### {labels[summary_key]}"])
    lines.append(f"- raw_count: {stats['raw_count']}")
    lines.append(f"- trimmed_count: {stats['trimmed_count']}")
    lines.append(f"- mean(us): {format_float(stats['mean'])}")
    lines.append(f"- min(us): {format_float(stats['min'])}")
    lines.append(f"- max(us): {format_float(stats['max'])}")
    lines.append(f"- p75(us): {format_float(stats['p75'])}")
    lines.append(f"- p90(us): {format_float(stats['p90'])}")
    lines.append(f"- p99(us): {format_float(stats['p99'])}")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    csv_path = Path(args.kernel_details_csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"kernel_details.csv 不存在: {csv_path}", file=sys.stderr)
        return 1

    side = infer_profile_side(csv_path)
    rows = load_rows(csv_path)
    stage_samples = build_stage_samples(rows, side)
    summaries = {name: summarize(values) for name, values in stage_samples.items()}

    payload = {
        "csv_path": str(csv_path),
        "side": side,
        "extreme_trim_rule": "drop_smallest_and_largest_one_sample_if_possible",
        "summaries": summaries,
    }

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(csv_path, side, summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
