#!/usr/bin/env python3
"""Summarize AFD attention/ffn kernel durations from Ascend kernel_details.csv files."""

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
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


ROLE_DIR_TO_NAME = {
    "modelrunner": "attn",
    "ffn": "ffn",
}

DEFAULT_PROFILE_NAME = "DEFAULT_PROFILE"


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


def percentile_field_name(percentile_value: float) -> str:
    if percentile_value.is_integer():
        label = str(int(percentile_value))
    else:
        label = str(percentile_value).replace(".", "_")
    return f"p{label}_us"


def summarize(values: List[float], percentiles: Sequence[float]) -> Dict[str, Optional[float]]:
    stats: Dict[str, Optional[float]] = {
        "count": len(values),
        "mean_us": (sum(values) / len(values)) if values else None,
    }
    for percentile_value in percentiles:
        stats[percentile_field_name(percentile_value)] = percentile(
            values,
            percentile_value / 100.0,
        )
    return stats


def parse_op_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_percentiles(text: Optional[str]) -> List[float]:
    values: List[float] = []
    for part in parse_op_list(text):
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"非法分位数: {part}") from exc
        if value < 0 or value > 100:
            raise ValueError(f"分位数必须位于 0 到 100 之间: {part}")
        values.append(value)
    if not values:
        raise ValueError("请至少指定一个分位数")
    return sorted(set(values))


def find_role_index(parts: Sequence[str]) -> Optional[int]:
    for index in range(len(parts) - 1, -1, -1):
        if normalize(parts[index]) in ROLE_DIR_TO_NAME:
            return index
    return None


def infer_role_from_path(csv_path: Path) -> Optional[str]:
    role_index = find_role_index(csv_path.parts)
    if role_index is None:
        return None
    return ROLE_DIR_TO_NAME[normalize(csv_path.parts[role_index])]


def discover_kernel_detail_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.name != "kernel_details.csv":
            return []
        return [input_path] if infer_role_from_path(input_path) else []
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("kernel_details.csv")
            if path.is_file() and infer_role_from_path(path)
        )
    return []


def op_matches_name(kernel_name: str, expected_name: str) -> bool:
    normalized_kernel = normalize(kernel_name)
    normalized_expected = normalize(expected_name)
    return (
        normalized_kernel == normalized_expected
        or normalized_kernel.startswith(normalized_expected)
    )


def infer_context(csv_path: Path) -> Tuple[str, str, str, str]:
    parts = list(csv_path.parts)
    role_index = find_role_index(parts)
    if role_index is None:
        return ("UNKNOWN_EXPERIMENT", "UNKNOWN_PROFILE", "UNKNOWN_RANK", "unknown")

    role = ROLE_DIR_TO_NAME[normalize(parts[role_index])]

    profile_index: Optional[int] = None
    for index in range(role_index - 1, -1, -1):
        if normalize(parts[index]) == "profile":
            profile_index = index
            break

    experiment = "UNKNOWN_EXPERIMENT"
    profile_name = "UNKNOWN_PROFILE"
    if profile_index is not None:
        if profile_index >= 1:
            experiment = parts[profile_index - 1]
        if role_index == profile_index + 1:
            # Support flattened layouts like profile/model_runner/<rank>/...
            profile_name = DEFAULT_PROFILE_NAME
        elif role_index > profile_index + 1:
            profile_name = parts[profile_index + 1]

    rank_name = "UNKNOWN_RANK"
    if len(parts) > role_index + 1:
        rank_name = parts[role_index + 1]

    return (experiment, profile_name, rank_name, role)


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


def scan_single_csv(
    csv_path_str: str,
    requested_ops: Dict[str, List[str]],
) -> Dict[str, object]:
    csv_path = Path(csv_path_str)
    experiment, profile_name, rank_name, role = infer_context(csv_path)
    matched = load_matching_durations(csv_path, requested_ops.get(role, []))
    return {
        "csv_path": csv_path_str,
        "experiment": experiment,
        "profile_name": profile_name,
        "rank_name": rank_name,
        "role": role,
        "matched": matched,
    }


def collect_csv_results(
    csv_paths: Sequence[Path],
    requested_ops: Dict[str, List[str]],
    workers: int,
) -> List[Dict[str, object]]:
    if len(csv_paths) <= 1 or workers <= 1:
        return [scan_single_csv(str(csv_path), requested_ops) for csv_path in csv_paths]

    futures_args = [str(csv_path) for csv_path in csv_paths]

    def run_with_executor(
        executor_cls: type[concurrent.futures.Executor],
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        with executor_cls(max_workers=workers) as executor:
            futures = [
                executor.submit(scan_single_csv, csv_path_str, requested_ops)
                for csv_path_str in futures_args
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results

    try:
        results = run_with_executor(concurrent.futures.ProcessPoolExecutor)
    except (OSError, PermissionError):
        results = run_with_executor(concurrent.futures.ThreadPoolExecutor)

    results.sort(key=lambda item: str(item["csv_path"]))
    return results


def append_record(
    records: List[Dict[str, object]],
    percentile_fields: Sequence[str],
    percentiles: Sequence[float],
    scope: str,
    experiment: str,
    profile_name: str,
    rank_name: str,
    role: str,
    op_name: str,
    values: List[float],
    csv_count: int,
) -> None:
    stats = summarize(values, percentiles)
    record: Dict[str, object] = {
        "scope": scope,
        "experiment": experiment,
        "profile_name": profile_name,
        "rank_name": rank_name,
        "role": role,
        "op_name": op_name,
        "csv_count": csv_count,
        "sample_count": stats["count"],
        "mean_us": stats["mean_us"],
    }
    for field_name in percentile_fields:
        record[field_name] = stats[field_name]
    records.append(record)


def build_records(
    csv_paths: Sequence[Path],
    requested_ops: Dict[str, List[str]],
    workers: int,
    percentiles: Sequence[float],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    percentile_fields = [percentile_field_name(value) for value in percentiles]
    profile_samples: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    experiment_samples: DefaultDict[Tuple[str, str, str], List[float]] = defaultdict(list)
    overall_samples: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    profile_csvs: DefaultDict[Tuple[str, str, str, str], int] = defaultdict(int)
    experiment_csvs: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    overall_csvs: DefaultDict[str, int] = defaultdict(int)

    for result in collect_csv_results(csv_paths, requested_ops, workers):
        experiment = str(result["experiment"])
        profile_name = str(result["profile_name"])
        rank_name = str(result["rank_name"])
        role = str(result["role"])
        profile_csvs[(experiment, profile_name, rank_name, role)] += 1
        experiment_csvs[(experiment, role)] += 1
        overall_csvs[role] += 1
        matched = result["matched"]
        for op_name, durations in matched.items():
            if not durations:
                continue
            profile_samples[(experiment, profile_name, rank_name, role, op_name)].extend(durations)
            experiment_samples[(experiment, role, op_name)].extend(durations)
            overall_samples[(role, op_name)].extend(durations)

    for (experiment, profile_name, rank_name, role, op_name), values in sorted(profile_samples.items()):
        append_record(
            records=records,
            percentile_fields=percentile_fields,
            percentiles=percentiles,
            scope="profile",
            experiment=experiment,
            profile_name=profile_name,
            rank_name=rank_name,
            role=role,
            op_name=op_name,
            values=values,
            csv_count=profile_csvs[(experiment, profile_name, rank_name, role)],
        )

    for (experiment, role, op_name), values in sorted(experiment_samples.items()):
        append_record(
            records=records,
            percentile_fields=percentile_fields,
            percentiles=percentiles,
            scope="experiment",
            experiment=experiment,
            profile_name="ALL",
            rank_name="ALL",
            role=role,
            op_name=op_name,
            values=values,
            csv_count=experiment_csvs[(experiment, role)],
        )

    for (role, op_name), values in sorted(overall_samples.items()):
        append_record(
            records=records,
            percentile_fields=percentile_fields,
            percentiles=percentiles,
            scope="overall",
            experiment="ALL",
            profile_name="ALL",
            rank_name="ALL",
            role=role,
            op_name=op_name,
            values=values,
            csv_count=overall_csvs[role],
        )

    return records


def filter_records_by_scopes(
    records: Sequence[Dict[str, object]],
    scopes: Iterable[str],
) -> List[Dict[str, object]]:
    scope_set = set(scopes)
    return [record for record in records if str(record.get("scope")) in scope_set]


def sort_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    scope_order = {
        "overall": 0,
        "experiment": 1,
        "profile": 2,
    }
    return sorted(
        records,
        key=lambda record: (
            str(record.get("role") or ""),
            str(record.get("op_name") or ""),
            scope_order.get(str(record.get("scope") or ""), 99),
            str(record.get("experiment") or ""),
            str(record.get("profile_name") or ""),
            str(record.get("rank_name") or ""),
        ),
    )


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def format_csv_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def write_csv(
    output_path: Path,
    records: List[Dict[str, object]],
    percentile_fields: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "experiment",
        "profile_name",
        "rank_name",
        "role",
        "op_name",
        "csv_count",
        "sample_count",
        "mean_us",
        *percentile_fields,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sort_records(records):
            csv_record = dict(record)
            csv_record["mean_us"] = format_csv_float(csv_record.get("mean_us"))
            for field in percentile_fields:
                csv_record[field] = format_csv_float(csv_record.get(field))
            writer.writerow(csv_record)


def render_markdown(
    input_path: Path,
    output_path: Path,
    requested_ops: Dict[str, List[str]],
    csv_paths: Sequence[Path],
    workers: int,
    percentiles: Sequence[float],
    records: Sequence[Dict[str, object]],
) -> str:
    percentile_fields = [percentile_field_name(value) for value in percentiles]
    lines = [
        "## AFD Kernel Profile Summary",
        "",
        f"- 输入路径: `{input_path}`",
        f"- 命中 kernel_details.csv 数量: `{len(csv_paths)}`",
        f"- 并行进程数: `{workers}`",
        f"- 输出 CSV: `{output_path}`",
        f"- Attention 算子: `{', '.join(requested_ops['attn']) or '(none)'}`",
        f"- FFN 算子: `{', '.join(requested_ops['ffn']) or '(none)'}`",
        f"- 分位数: `{', '.join(str(int(value)) if value.is_integer() else str(value) for value in percentiles)}`",
    ]
    overall_records = [record for record in records if record["scope"] == "overall"]
    if overall_records:
        lines.append("")
        lines.append("### Overall")
        for record in overall_records:
            summary_parts = [
                f"{record['role']}/{record['op_name']}: count={record['sample_count']}",
                f"mean={format_float(record['mean_us'])} us",
            ]
            for field in percentile_fields:
                summary_parts.append(f"{field[:-3]}={format_float(record.get(field))} us")
            lines.append("- " + ", ".join(summary_parts))
    return "\n".join(lines)


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_afd_kernel_profile_summary.csv")
    prefix = input_path.name or "benchmark_result"
    return Path.cwd() / f"{prefix}_afd_kernel_profile_summary.csv"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归统计 AFD 场景下 attn/model_runner 与 ffn 两侧给定算子的耗时分位数。"
    )
    parser.add_argument("input_path", help="kernel_details.csv 路径，或包含多个实验结果的目录")
    parser.add_argument(
        "--attn-ops",
        default="",
        help="Attention 侧算子名列表，逗号分隔；会匹配 model_runner 下的 kernel_details.csv",
    )
    parser.add_argument(
        "--ffn-ops",
        default="",
        help="FFN 侧算子名列表，逗号分隔；会匹配 ffn 下的 kernel_details.csv",
    )
    parser.add_argument(
        "--percentiles",
        default="25,50,75,90,99",
        help="要输出的 P 分位数列表，逗号分隔，默认 25,50,75,90,99",
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

    requested_ops = {
        "attn": parse_op_list(args.attn_ops),
        "ffn": parse_op_list(args.ffn_ops),
    }
    if not requested_ops["attn"] and not requested_ops["ffn"]:
        print("请至少通过 --attn-ops 或 --ffn-ops 指定一侧算子", file=sys.stderr)
        return 1

    try:
        percentiles = parse_percentiles(args.percentiles)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    percentile_fields = [percentile_field_name(value) for value in percentiles]

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
        print(
            "未找到位于 model_runner 或 ffn 目录下的 kernel_details.csv: "
            f"{input_path}",
            file=sys.stderr,
        )
        return 1

    auto_workers = min(len(csv_paths), os.cpu_count() or 1)
    workers = args.workers if args.workers is not None else auto_workers
    workers = max(1, workers)

    all_records = build_records(csv_paths, requested_ops, workers, percentiles)
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
    write_csv(output_path, records, percentile_fields)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "attn_ops": requested_ops["attn"],
                    "ffn_ops": requested_ops["ffn"],
                    "percentiles": percentiles,
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
        print(
            render_markdown(
                input_path=input_path,
                output_path=output_path,
                requested_ops=requested_ops,
                csv_paths=csv_paths,
                workers=workers,
                percentiles=percentiles,
                records=records,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
