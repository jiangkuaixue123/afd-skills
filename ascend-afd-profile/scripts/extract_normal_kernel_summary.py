#!/usr/bin/env python3
"""Summarize normal-scene kernel durations from Ascend kernel_details.csv files."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple


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


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "count": len(values),
        "mean_us": (sum(values) / len(values)) if values else None,
        "p25_us": percentile(values, 0.25),
        "p50_us": percentile(values, 0.50),
        "p75_us": percentile(values, 0.75),
        "p90_us": percentile(values, 0.90),
        "p99_us": percentile(values, 0.99),
    }


def parse_op_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def discover_kernel_detail_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.name == "kernel_details.csv" else []
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("kernel_details.csv") if path.is_file())
    return []


def sanitize_field_suffix(op_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", op_name).strip("_")
    return sanitized or "op"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def format_csv_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def op_matches_name(kernel_name: str, expected_name: str) -> bool:
    normalized_kernel = normalize(kernel_name)
    normalized_expected = normalize(expected_name)
    return (
        normalized_kernel == normalized_expected
        or normalized_kernel.startswith(normalized_expected)
    )


def infer_context(csv_path: Path) -> Tuple[str, str, str]:
    parts = list(csv_path.parts)
    try:
        profile_index = parts.index("profile")
    except ValueError:
        return ("UNKNOWN_EXPERIMENT", "UNKNOWN_RANK", csv_path.parent.name)

    experiment = parts[profile_index - 1] if profile_index >= 1 else "UNKNOWN_EXPERIMENT"
    rank_name = parts[profile_index + 1] if len(parts) > profile_index + 1 else "UNKNOWN_RANK"
    profile_name = csv_path.parents[1].name if len(csv_path.parents) >= 2 else csv_path.stem
    return (experiment, rank_name, profile_name)


def load_matching_durations(csv_path: Path, op_names: Sequence[str]) -> Dict[str, List[float]]:
    matches: Dict[str, List[float]] = {op_name: [] for op_name in op_names}
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel_name = (row.get("Name") or row.get("OP Type") or row.get("Op Name") or "").strip()
            if not kernel_name:
                continue
            duration_us = parse_float(row.get("Duration(us)"))
            if duration_us is None:
                continue
            for op_name in op_names:
                if op_matches_name(kernel_name, op_name):
                    matches[op_name].append(duration_us)
    return matches


def scan_single_csv(csv_path_str: str, op_names: Sequence[str]) -> Dict[str, object]:
    csv_path = Path(csv_path_str)
    experiment, rank_name, profile_name = infer_context(csv_path)
    return {
        "csv_path": csv_path_str,
        "experiment": experiment,
        "rank_name": rank_name,
        "profile_name": profile_name,
        "matched": load_matching_durations(csv_path, op_names),
    }


def append_record(
    records: List[Dict[str, object]],
    scope: str,
    experiment: str,
    rank_name: str,
    profile_name: str,
    op_name: str,
    values: List[float],
    csv_count: int,
) -> None:
    stats = summarize(values)
    records.append(
        {
            "scope": scope,
            "experiment": experiment,
            "rank_name": rank_name,
            "profile_name": profile_name,
            "op_name": op_name,
            "csv_count": csv_count,
            "sample_count": stats["count"],
            "mean_us": stats["mean_us"],
            "p25_us": stats["p25_us"],
            "p50_us": stats["p50_us"],
            "p75_us": stats["p75_us"],
            "p90_us": stats["p90_us"],
            "p99_us": stats["p99_us"],
        }
    )


def collect_csv_results(
    csv_paths: Sequence[Path],
    op_names: Sequence[str],
    workers: int,
) -> List[Dict[str, object]]:
    if len(csv_paths) <= 1 or workers <= 1:
        return [scan_single_csv(str(csv_path), op_names) for csv_path in csv_paths]

    futures_args = [
        (str(csv_path), list(op_names))
        for csv_path in csv_paths
    ]

    def run_with_executor(
        executor_cls: type[concurrent.futures.Executor],
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        with executor_cls(max_workers=workers) as executor:
            futures = [
                executor.submit(scan_single_csv, csv_path_str, op_names_list)
                for csv_path_str, op_names_list in futures_args
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results

    try:
        results = run_with_executor(concurrent.futures.ProcessPoolExecutor)
    except (OSError, PermissionError):
        # Some restricted environments disallow process pools; fall back to threads.
        results = run_with_executor(concurrent.futures.ThreadPoolExecutor)

    results.sort(key=lambda item: str(item["csv_path"]))
    return results


def build_records(
    csv_paths: Sequence[Path],
    op_names: Sequence[str],
    workers: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    profile_samples: DefaultDict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    experiment_samples: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    overall_samples: DefaultDict[str, List[float]] = defaultdict(list)
    profile_csvs: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    experiment_csvs: DefaultDict[str, int] = defaultdict(int)

    for result in collect_csv_results(csv_paths, op_names, workers):
        experiment = str(result["experiment"])
        rank_name = str(result["rank_name"])
        profile_name = str(result["profile_name"])
        profile_csvs[(experiment, rank_name, profile_name)] += 1
        experiment_csvs[experiment] += 1
        matched = result["matched"]
        for op_name, durations in matched.items():
            if not durations:
                continue
            profile_samples[(experiment, rank_name, profile_name, op_name)].extend(durations)
            experiment_samples[(experiment, op_name)].extend(durations)
            overall_samples[op_name].extend(durations)

    for (experiment, rank_name, profile_name, op_name), values in sorted(profile_samples.items()):
        append_record(
            records=records,
            scope="profile",
            experiment=experiment,
            rank_name=rank_name,
            profile_name=profile_name,
            op_name=op_name,
            values=values,
            csv_count=profile_csvs[(experiment, rank_name, profile_name)],
        )

    for (experiment, op_name), values in sorted(experiment_samples.items()):
        append_record(
            records=records,
            scope="experiment",
            experiment=experiment,
            rank_name="ALL",
            profile_name="ALL",
            op_name=op_name,
            values=values,
            csv_count=experiment_csvs[experiment],
        )

    for op_name, values in sorted(overall_samples.items()):
        append_record(
            records=records,
            scope="overall",
            experiment="ALL",
            rank_name="ALL",
            profile_name="ALL",
            op_name=op_name,
            values=values,
            csv_count=len(csv_paths),
        )

    return records


def filter_records_by_scopes(
    records: Sequence[Dict[str, object]],
    scopes: Set[str],
) -> List[Dict[str, object]]:
    return [record for record in records if str(record.get("scope")) in scopes]


def sort_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    scope_order = {
        "overall": 0,
        "experiment": 1,
        "profile": 2,
    }
    return sorted(
        records,
        key=lambda record: (
            str(record.get("op_name") or ""),
            scope_order.get(str(record.get("scope") or ""), 99),
            str(record.get("experiment") or ""),
            str(record.get("rank_name") or ""),
            str(record.get("profile_name") or ""),
        ),
    )


def write_csv(output_path: Path, records: List[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "experiment",
        "rank_name",
        "profile_name",
        "op_name",
        "csv_count",
        "sample_count",
        "mean_us",
        "p25_us",
        "p50_us",
        "p75_us",
        "p90_us",
        "p99_us",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sort_records(records):
            csv_record = dict(record)
            for field in ("mean_us", "p25_us", "p50_us", "p75_us", "p90_us", "p99_us"):
                csv_record[field] = format_csv_float(csv_record.get(field))
            writer.writerow(csv_record)


def render_markdown(
    input_path: Path,
    output_path: Path,
    op_names: Sequence[str],
    csv_paths: Sequence[Path],
    workers: int,
    records: Sequence[Dict[str, object]],
) -> str:
    lines = [
        "## Normal Kernel Summary",
        "",
        f"- 输入路径: `{input_path}`",
        f"- 命中 kernel_details.csv 数量: `{len(csv_paths)}`",
        f"- 并行进程数: `{workers}`",
        f"- 输出 CSV: `{output_path}`",
        f"- 统计算子: `{', '.join(op_names)}`",
    ]
    overall_records = [record for record in records if record["scope"] == "overall"]
    if overall_records:
        lines.append("")
        lines.append("### Overall")
        for record in overall_records:
            lines.append(
                "- "
                f"{record['op_name']}: count={record['sample_count']}, "
                f"mean={format_float(record['mean_us'])} us, "
                f"p50={format_float(record['p50_us'])} us, "
                f"p90={format_float(record['p90_us'])} us, "
                f"p99={format_float(record['p99_us'])} us"
            )
    return "\n".join(lines)


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_normal_kernel_summary.csv")
    prefix = input_path.name or "benchmark_result"
    return Path.cwd() / f"{prefix}_normal_kernel_summary.csv"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归统计 normal 场景下给定算子在所有 kernel_details.csv 中的耗时分位数。"
    )
    parser.add_argument("input_path", help="kernel_details.csv 路径，或包含多个实验结果的目录")
    parser.add_argument(
        "--ops",
        required=True,
        help="要统计的算子名列表，逗号分隔；按名称规范化后做 exact/prefix 匹配",
    )
    parser.add_argument("-o", "--output", help="输出 CSV 路径；未指定时自动生成")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="终端输出格式，默认 markdown",
    )
    parser.add_argument(
        "--scopes",
        default="profile,experiment,overall",
        help="输出范围，逗号分隔；可选值: profile,experiment,overall",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数；未指定时自动选择，传 1 可关闭多进程",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}", file=sys.stderr)
        return 1

    op_names = parse_op_list(args.ops)
    if not op_names:
        print("请通过 --ops 指定至少一个算子名", file=sys.stderr)
        return 1

    scopes = set(parse_op_list(args.scopes))
    valid_scopes = {"profile", "experiment", "overall"}
    invalid_scopes = scopes - valid_scopes
    if not scopes:
        print("请通过 --scopes 指定至少一个输出范围", file=sys.stderr)
        return 1
    if invalid_scopes:
        print(
            f"存在不支持的 scope: {', '.join(sorted(invalid_scopes))}；"
            f"仅支持 {', '.join(sorted(valid_scopes))}",
            file=sys.stderr,
        )
        return 1

    csv_paths = discover_kernel_detail_files(input_path)
    if not csv_paths:
        print(f"未找到 kernel_details.csv: {input_path}", file=sys.stderr)
        return 1

    auto_workers = min(len(csv_paths), os.cpu_count() or 1)
    workers = args.workers if args.workers is not None else auto_workers
    workers = max(1, workers)

    all_records = build_records(csv_paths, op_names, workers)
    records = filter_records_by_scopes(all_records, scopes)
    if not all_records:
        print("给定算子列表没有命中任何 kernel 记录", file=sys.stderr)
        return 1
    if not records:
        print("给定 --scopes 过滤后没有可输出的记录", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )
    write_csv(output_path, records)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "ops": op_names,
                    "scopes": sorted(scopes),
                    "csv_count": len(csv_paths),
                    "workers": workers,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_markdown(input_path, output_path, op_names, csv_paths, workers, records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
