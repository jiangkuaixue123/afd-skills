#!/usr/bin/env python3
"""Summarize per-microbatch stage durations from Ascend kernel_details.csv."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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


def op_matches_name(op_name: str, expected_name: str) -> bool:
    normalized_op = normalize(op_name)
    normalized_expected = normalize(expected_name)
    return normalized_op == normalized_expected or normalized_op.startswith(normalized_expected)


def infer_profile_side(csv_path: Path) -> str:
    normalized_parts = [normalize(part) for part in csv_path.parts]
    if "attention" in normalized_parts:
        return "attention"
    if "modelrunner" in normalized_parts:
        return "attention"
    if "ffn" in normalized_parts:
        return "ffn"
    raise ValueError(
        "无法从路径推断侧别，请确保 kernel_details.csv 位于 profile/attention、profile/model_runner 或 profile/ffn 目录下。"
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


def remove_large_outliers(values: List[float]) -> List[float]:
    if len(values) < 4:
        return list(values)
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    if q1 is None or q3 is None:
        return list(values)
    iqr = q3 - q1
    if iqr <= 0:
        return list(values)
    upper_fence = q3 + 3 * iqr
    filtered = [value for value in values if value <= upper_fence]
    return filtered or list(values)


def mean_with_outlier_filter(values: List[float]) -> Optional[float]:
    filtered = remove_large_outliers(values)
    trimmed = trim_extremes(filtered)
    if not trimmed:
        return None
    return sum(trimmed) / len(trimmed)


def stage_label(side: str) -> str:
    if side == "attention":
        return "attn_e2a_to_a2e_us"
    if side == "ffn":
        return "ffn_a2e_to_e2a_us"
    raise ValueError(f"未知侧别: {side}")


def build_stage_samples(rows: List[KernelRow], side: str) -> Dict[str, List[float]]:
    stage_name = stage_label(side)
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


def build_op_samples(rows: List[KernelRow], op_names: Sequence[str]) -> Dict[str, List[float]]:
    op_samples: Dict[str, List[float]] = {op_name: [] for op_name in op_names}
    for row in rows:
        for op_name in op_names:
            if op_matches_name(row.name, op_name):
                op_samples[op_name].append(row.duration_us)
    return op_samples


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    filtered = remove_large_outliers(values)
    trimmed = trim_extremes(filtered)
    if not trimmed:
        return {
            "raw_count": len(values),
            "trimmed_count": len(trimmed),
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
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
        "p50": percentile(trimmed, 0.50),
        "p75": percentile(trimmed, 0.75),
        "p90": percentile(trimmed, 0.90),
        "p99": percentile(trimmed, 0.99),
    }


def parse_op_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def sanitize_field_suffix(op_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", op_name).strip("_")
    return sanitized or "op"


def op_mean_field_name(op_name: str) -> str:
    return f"{sanitize_field_suffix(op_name)}_mean_us"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def discover_kernel_detail_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("kernel_details.csv") if path.is_file())
    return []


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_stage_summary.csv")
    return input_path / "kernel_stage_summary.csv"


def looks_like_experiment_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (
            (path / "profile" / "attention").is_dir()
            or (path / "profile" / "model_runner").is_dir()
        )
        and (path / "profile" / "ffn").is_dir()
    )


def profile_side_dir(experiment_dir: Path, side: str) -> Path:
    if side == "attention":
        attention_dir = experiment_dir / "profile" / "attention"
        if attention_dir.is_dir():
            return attention_dir
        model_runner_dir = experiment_dir / "profile" / "model_runner"
        if model_runner_dir.is_dir():
            return model_runner_dir
    if side == "ffn":
        return experiment_dir / "profile" / "ffn"
    raise ValueError(f"未知侧别: {side}")


def discover_experiment_dirs(benchmark_root: Path) -> List[Path]:
    if not benchmark_root.is_dir():
        return []
    return sorted(path for path in benchmark_root.iterdir() if looks_like_experiment_dir(path))


def default_benchmark_output_path(benchmark_root: Path) -> Path:
    prefix = benchmark_root.name or "benchmark_result"
    cwd = Path.cwd()
    return cwd / f"{prefix}_kernel_stage_summary.csv"


def infer_rank_name(csv_path: Path) -> str:
    # .../<rank_dir>/ASCEND_PROFILER_OUTPUT/kernel_details.csv
    if len(csv_path.parents) >= 2:
        return csv_path.parents[1].name
    return csv_path.stem


def summarize_csv(
    csv_path: Path,
    side_ops: Dict[str, List[str]],
) -> Dict[str, object]:
    side = infer_profile_side(csv_path)
    rows = load_rows(csv_path)
    stage_samples = build_stage_samples(rows, side)
    op_samples = build_op_samples(rows, side_ops.get(side, []))
    summary_key = stage_label(side)
    summaries = {name: summarize(values) for name, values in stage_samples.items()}
    op_means = {
        op_name: mean_with_outlier_filter(values)
        for op_name, values in op_samples.items()
    }
    return {
        "csv_path": str(csv_path),
        "side": side,
        "rank_name": infer_rank_name(csv_path),
        "extreme_trim_rule": "drop_smallest_and_largest_one_sample_if_possible",
        "stage_samples": stage_samples,
        "op_samples": op_samples,
        "op_means": op_means,
        "summaries": summaries,
        "summary_key": summary_key,
    }


def build_output_records(payloads: List[Dict[str, object]], side_ops: Dict[str, List[str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    grouped_samples: Dict[Tuple[str, str], List[float]] = {}
    grouped_payloads: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    grouped_op_samples: Dict[Tuple[str, str], List[float]] = {}

    for payload in payloads:
        summaries = payload["summaries"]
        for stage_name, stats in summaries.items():
            samples = payload["stage_samples"].get(stage_name, [])
            key = (payload["side"], stage_name)
            grouped_samples.setdefault(key, []).extend(samples)
            grouped_payloads.setdefault(key, []).append(payload)
            records.append(
                {
                    "scope": "rank",
                    "rank_name": payload["rank_name"],
                    "csv_path": payload["csv_path"],
                    "side": payload["side"],
                    "stage_name": stage_name,
                    "rank_count": 1,
                    "raw_count": stats["raw_count"],
                    "trimmed_count": stats["trimmed_count"],
                    "mean_us": stats["mean"],
                    "min_us": stats["min"],
                    "max_us": stats["max"],
                    "p50_us": stats["p50"],
                    "p75_us": stats["p75"],
                    "p90_us": stats["p90"],
                    "p99_us": stats["p99"],
                    "trim_rule": payload["extreme_trim_rule"],
                }
            )
            for op_name in side_ops.get(payload["side"], []):
                grouped_op_samples.setdefault((payload["side"], op_name), []).extend(
                    payload["op_samples"].get(op_name, [])
                )
                records[-1][op_mean_field_name(op_name)] = payload["op_means"].get(op_name)

    for (side, stage_name), samples in sorted(grouped_samples.items()):
        payload_group = grouped_payloads[(side, stage_name)]
        if len(payload_group) <= 1:
            continue
        stats = summarize(samples)
        records.append(
            {
                "scope": "overall",
                "rank_name": "ALL",
                "csv_path": "",
                "side": side,
                "stage_name": stage_name,
                "rank_count": len(payload_group),
                "raw_count": stats["raw_count"],
                "trimmed_count": stats["trimmed_count"],
                "mean_us": stats["mean"],
                "min_us": stats["min"],
                "max_us": stats["max"],
                "p50_us": stats["p50"],
                "p75_us": stats["p75"],
                "p90_us": stats["p90"],
                "p99_us": stats["p99"],
                "trim_rule": payload_group[0]["extreme_trim_rule"],
            }
        )
        for op_name in side_ops.get(side, []):
            op_values = grouped_op_samples.get((side, op_name), [])
            records[-1][op_mean_field_name(op_name)] = (
                mean_with_outlier_filter(op_values)
            )

    return records


def summarize_payload_group(
    payloads: List[Dict[str, object]],
    side: str,
    side_ops: Dict[str, List[str]],
) -> Optional[Dict[str, object]]:
    if not payloads:
        return None

    stage_name = stage_label(side)
    stage_values: List[float] = []
    for payload in payloads:
        stage_values.extend(payload["stage_samples"].get(stage_name, []))
    stats = summarize(stage_values)
    record: Dict[str, object] = {
        "experiment": "",
        "side": side,
        "rank_count": len(payloads),
        "raw_count": stats["raw_count"],
        "trimmed_count": stats["trimmed_count"],
        "mean_us": stats["mean"],
        "min_us": stats["min"],
        "max_us": stats["max"],
        "p50_us": stats["p50"],
        "p75_us": stats["p75"],
        "p90_us": stats["p90"],
        "p99_us": stats["p99"],
        "trim_rule": payloads[0]["extreme_trim_rule"],
    }
    for op_name in side_ops.get(side, []):
        op_values: List[float] = []
        for payload in payloads:
            op_values.extend(payload["op_samples"].get(op_name, []))
        record[op_mean_field_name(op_name)] = (
            mean_with_outlier_filter(op_values)
        )
    return record


def build_benchmark_side_records(
    benchmark_root: Path,
    side: str,
    side_ops: Dict[str, List[str]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for experiment_dir in discover_experiment_dirs(benchmark_root):
        side_dir = profile_side_dir(experiment_dir, side)
        csv_paths = discover_kernel_detail_files(side_dir)
        payloads = [summarize_csv(csv_path, side_ops) for csv_path in csv_paths]
        record = summarize_payload_group(payloads, side, side_ops)
        if record is None:
            continue
        record["experiment"] = experiment_dir.name
        records.append(record)
    return records


def process_experiment_side(
    experiment_dir_str: str,
    side: str,
    side_ops: Dict[str, List[str]],
) -> Optional[Dict[str, object]]:
    experiment_dir = Path(experiment_dir_str)
    side_dir = profile_side_dir(experiment_dir, side)
    csv_paths = discover_kernel_detail_files(side_dir)
    payloads = [summarize_csv(csv_path, side_ops) for csv_path in csv_paths]
    record = summarize_payload_group(payloads, side, side_ops)
    if record is None:
        return None
    record["experiment"] = experiment_dir.name
    return record


def build_benchmark_side_records_parallel(
    benchmark_root: Path,
    side: str,
    side_ops: Dict[str, List[str]],
    workers: int,
) -> List[Dict[str, object]]:
    experiment_dirs = discover_experiment_dirs(benchmark_root)
    if len(experiment_dirs) <= 1 or workers <= 1:
        return build_benchmark_side_records(benchmark_root, side, side_ops)

    records: List[Dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_experiment_side, str(experiment_dir), side, side_ops)
            for experiment_dir in experiment_dirs
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            if record is not None:
                records.append(record)
    records.sort(key=lambda item: item["experiment"])
    return records


def write_csv(output_path: Path, records: List[Dict[str, object]], side_ops: Dict[str, List[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "rank_name",
        "csv_path",
        "side",
        "stage_name",
        "rank_count",
        "raw_count",
        "trimmed_count",
        "mean_us",
        "min_us",
        "max_us",
        "p50_us",
        "p75_us",
        "p90_us",
        "p99_us",
        "trim_rule",
    ]
    for side in ("attention", "ffn"):
        for op_name in side_ops.get(side, []):
            fieldnames.append(op_mean_field_name(op_name))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_benchmark_csv(output_path: Path, records: List[Dict[str, object]], side_ops: Dict[str, List[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "side",
        "rank_count",
        "raw_count",
        "trimmed_count",
        "mean_us",
        "min_us",
        "max_us",
        "p50_us",
        "p75_us",
        "p90_us",
        "p99_us",
        "trim_rule",
    ]
    for side in ("attention", "ffn"):
        for op_name in side_ops.get(side, []):
            fieldnames.append(op_mean_field_name(op_name))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 kernel_details.csv 统计 AFD Attention/FFN 侧每个 microbatch 的阶段耗时，并写入 CSV。"
    )
    parser.add_argument(
        "input_path",
        help="kernel_details.csv 路径，或包含多个 kernel_details.csv 的目录",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出 CSV 路径；未指定时自动生成",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="终端输出格式，默认 markdown",
    )
    parser.add_argument(
        "--attn-ops",
        default="",
        help="Attention 侧需要额外统计平均时延的算子名列表，逗号分隔",
    )
    parser.add_argument(
        "--ffn-ops",
        default="",
        help="FFN 侧需要额外统计平均时延的算子名列表，逗号分隔",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="benchmark_result 模式下的并行进程数；未指定时自动选择",
    )
    return parser.parse_args(argv)


def render_markdown(
    payloads: List[Dict[str, object]],
    records: List[Dict[str, object]],
    output_path: Path,
) -> str:
    labels = {
        "attn_e2a_to_a2e_us": "Attention 侧 microbatch 执行时间（E2a -> 下一次 A2e 之间，不含 marker）",
        "ffn_a2e_to_e2a_us": "FFN 侧 microbatch 执行时间（A2e -> 下一次 E2a 之间，不含 marker）",
    }
    lines = [
        "## Kernel Stage Summary",
        "",
        f"- 输入数量: `{len(payloads)}`",
        f"- CSV 输出: `{output_path}`",
        "- 极值处理: 每组样本默认去掉 1 个最小值和 1 个最大值",
    ]
    requested_ops = {
        side: list(dict.fromkeys(
            op_name
            for payload in payloads
            if payload["side"] == side
            for op_name in payload["op_means"].keys()
        ))
        for side in ("attention", "ffn")
    }
    if requested_ops["attention"]:
        lines.append(f"- Attention 算子均值统计: `{', '.join(requested_ops['attention'])}`")
    if requested_ops["ffn"]:
        lines.append(f"- FFN 算子均值统计: `{', '.join(requested_ops['ffn'])}`")
    overall_records = [record for record in records if record["scope"] == "overall"]
    if overall_records:
        lines.append(f"- 汇总行数量: `{len(overall_records)}`")
        for record in overall_records:
            lines.extend(
                [
                    "",
                    f"### Overall `{record['side']}` / `{record['stage_name']}`",
                    f"- rank_count: {record['rank_count']}",
                    f"- raw_count: {record['raw_count']}",
                    f"- trimmed_count: {record['trimmed_count']}",
                    f"- mean(us): {format_float(record['mean_us'])}",
                    f"- min(us): {format_float(record['min_us'])}",
                    f"- max(us): {format_float(record['max_us'])}",
                    f"- p50(us): {format_float(record['p50_us'])}",
                    f"- p75(us): {format_float(record['p75_us'])}",
                    f"- p90(us): {format_float(record['p90_us'])}",
                    f"- p99(us): {format_float(record['p99_us'])}",
                ]
            )
            for op_name in requested_ops.get(record["side"], []):
                field = op_mean_field_name(op_name)
                lines.append(f"- op_mean({op_name})(us): {format_float(record.get(field))}")

    for payload in payloads:
        summary_key = payload["summary_key"]
        stats = payload["summaries"][summary_key]
        lines.extend(
            [
                "",
                f"### {labels[summary_key]}",
                f"- rank: `{payload['rank_name']}`",
                f"- 输入文件: `{payload['csv_path']}`",
                f"- 侧别: `{payload['side']}`",
                f"- raw_count: {stats['raw_count']}",
                f"- trimmed_count: {stats['trimmed_count']}",
                f"- mean(us): {format_float(stats['mean'])}",
                f"- min(us): {format_float(stats['min'])}",
                f"- max(us): {format_float(stats['max'])}",
                f"- p50(us): {format_float(stats['p50'])}",
                f"- p75(us): {format_float(stats['p75'])}",
                f"- p90(us): {format_float(stats['p90'])}",
                f"- p99(us): {format_float(stats['p99'])}",
            ]
        )
        for op_name, value in payload["op_means"].items():
            lines.append(f"- op_mean({op_name})(us): {format_float(value)}")
    return "\n".join(lines)


def render_benchmark_markdown(
    benchmark_root: Path,
    output_path: Path,
    side_records: Dict[str, List[Dict[str, object]]],
    side_ops: Dict[str, List[str]],
) -> str:
    lines = [
        "## Benchmark Kernel Stage Summary",
        "",
        f"- benchmark_result: `{benchmark_root}`",
        f"- merged CSV: `{output_path}`",
    ]
    if side_ops["attention"]:
        lines.append(f"- Attention 算子均值统计: `{', '.join(side_ops['attention'])}`")
    if side_ops["ffn"]:
        lines.append(f"- FFN 算子均值统计: `{', '.join(side_ops['ffn'])}`")
    for side in ("attention", "ffn"):
        lines.extend(
            [
                "",
                f"### {side}",
                f"- 实验数: {len(side_records[side])}",
            ]
        )
        for record in side_records[side]:
            lines.extend(
                [
                    f"- {record['experiment']}: mean={format_float(record['mean_us'])} us, p90={format_float(record['p90_us'])} us, rank_count={record['rank_count']}",
                ]
            )
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}", file=sys.stderr)
        return 1

    side_ops = {
        "attention": parse_op_list(args.attn_ops),
        "ffn": parse_op_list(args.ffn_ops),
    }

    experiment_dirs = discover_experiment_dirs(input_path)
    if experiment_dirs:
        output_path = default_benchmark_output_path(input_path)
        auto_workers = min(len(experiment_dirs), os.cpu_count() or 1)
        workers = args.workers if args.workers is not None else auto_workers
        workers = max(1, workers)
        side_records = {
            "attention": build_benchmark_side_records_parallel(input_path, "attention", side_ops, workers),
            "ffn": build_benchmark_side_records_parallel(input_path, "ffn", side_ops, workers),
        }
        merged_records = sorted(
            side_records["attention"] + side_records["ffn"],
            key=lambda item: (item["experiment"], item["side"]),
        )
        write_benchmark_csv(output_path, merged_records, side_ops)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "mode": "benchmark_result",
                        "benchmark_result": str(input_path),
                        "requested_ops": side_ops,
                        "workers": workers,
                        "output": str(output_path),
                        "records": merged_records,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(render_benchmark_markdown(input_path, output_path, side_records, side_ops))
        return 0

    csv_paths = discover_kernel_detail_files(input_path)
    if not csv_paths:
        print(
            f"未找到 kernel_details.csv: {input_path}",
            file=sys.stderr,
        )
        return 1

    payloads = [summarize_csv(csv_path, side_ops) for csv_path in csv_paths]
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)
    records = build_output_records(payloads, side_ops)
    write_csv(output_path, records, side_ops)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "csv_output": str(output_path),
                    "requested_ops": side_ops,
                    "results": payloads,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_markdown(payloads, records, output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
