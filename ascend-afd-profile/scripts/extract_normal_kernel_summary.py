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
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_PROFILE_NAME = "DEFAULT_PROFILE"
ROLE_DIR_NAMES = {"attention", "ffn", "modelrunner"}


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


def parse_ops_file(path: Path) -> List[str]:
    items: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            items.extend(parse_op_list(stripped))
    return items


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


def discover_kernel_detail_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.name == "kernel_details.csv" else []
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("kernel_details.csv") if path.is_file())
    return []


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


def find_last_index(parts: Sequence[str], normalized_name: str) -> Optional[int]:
    for index in range(len(parts) - 1, -1, -1):
        if normalize(parts[index]) == normalized_name:
            return index
    return None


def find_last_role_index(parts: Sequence[str], profile_index: int) -> Optional[int]:
    for index in range(len(parts) - 1, profile_index, -1):
        if normalize(parts[index]) in ROLE_DIR_NAMES:
            return index
    return None


def infer_context(csv_path: Path) -> Tuple[str, str, str]:
    parts = list(csv_path.parts)
    profile_index = find_last_index(parts, "profile")
    if profile_index is None:
        return ("UNKNOWN_EXPERIMENT", "UNKNOWN_PROFILE", "UNKNOWN_RANK")

    experiment = parts[profile_index - 1] if profile_index >= 1 else "UNKNOWN_EXPERIMENT"
    role_index = find_last_role_index(parts, profile_index)

    profile_name = "UNKNOWN_PROFILE"
    rank_name = "UNKNOWN_RANK"

    if role_index is None:
        if len(parts) > profile_index + 1:
            profile_name = parts[profile_index + 1]
        if len(parts) > profile_index + 2:
            rank_name = parts[profile_index + 2]
        elif len(csv_path.parents) >= 2:
            rank_name = csv_path.parents[1].name
        return (experiment, profile_name, rank_name)

    if role_index == profile_index + 1:
        profile_name = DEFAULT_PROFILE_NAME
    elif len(parts) > profile_index + 1:
        profile_name = parts[profile_index + 1]

    if len(parts) > role_index + 1:
        rank_name = parts[role_index + 1]
    elif len(csv_path.parents) >= 2:
        rank_name = csv_path.parents[1].name

    return (experiment, profile_name, rank_name)


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


def load_loop_durations(csv_path: Path, ordered_ops: Sequence[str]) -> List[float]:
    if not ordered_ops:
        return []

    loop_totals: List[float] = []
    matched_count = 0
    running_sum = 0.0

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel_name = (row.get("Name") or row.get("OP Type") or row.get("Op Name") or "").strip()
            if not kernel_name:
                continue
            duration_us = parse_float(row.get("Duration(us)"))
            if duration_us is None:
                continue

            if matched_count == 0:
                if op_matches_name(kernel_name, ordered_ops[0]):
                    matched_count = 1
                    running_sum = duration_us
                continue

            expected_name = ordered_ops[matched_count]
            if op_matches_name(kernel_name, expected_name):
                matched_count += 1
                running_sum += duration_us
                if matched_count == len(ordered_ops):
                    loop_totals.append(running_sum)
                    matched_count = 0
                    running_sum = 0.0
            elif op_matches_name(kernel_name, ordered_ops[0]):
                matched_count = 1
                running_sum = duration_us

    return loop_totals


def load_samples(
    csv_path: Path,
    match_mode: str,
    op_names: Sequence[str],
    loop_name: str,
) -> Dict[str, List[float]]:
    if match_mode == "loop":
        return {loop_name: load_loop_durations(csv_path, op_names)}
    return load_matching_durations(csv_path, op_names)


def scan_single_csv(
    csv_path_str: str,
    match_mode: str,
    op_names: Sequence[str],
    loop_name: str,
) -> Dict[str, object]:
    csv_path = Path(csv_path_str)
    experiment, profile_name, rank_name = infer_context(csv_path)
    return {
        "csv_path": csv_path_str,
        "experiment": experiment,
        "profile_name": profile_name,
        "rank_name": rank_name,
        "matched": load_samples(csv_path, match_mode, op_names, loop_name),
    }


def scan_single_csv_safe(
    csv_path_str: str,
    match_mode: str,
    op_names: Sequence[str],
    loop_name: str,
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, str]]]:
    try:
        return scan_single_csv(csv_path_str, match_mode, op_names, loop_name), None
    except Exception as exc:
        csv_path = Path(csv_path_str)
        return None, {
            "csv_path": csv_path_str,
            "directory": str(csv_path.parent),
            "error": str(exc),
        }


def collect_csv_results(
    csv_paths: Sequence[Path],
    match_mode: str,
    op_names: Sequence[str],
    loop_name: str,
    workers: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    if len(csv_paths) <= 1 or workers <= 1:
        results: List[Dict[str, object]] = []
        skipped: List[Dict[str, str]] = []
        for csv_path in csv_paths:
            result, error = scan_single_csv_safe(str(csv_path), match_mode, op_names, loop_name)
            if result is not None:
                results.append(result)
            elif error is not None:
                skipped.append(error)
        results.sort(key=lambda item: str(item["csv_path"]))
        skipped.sort(key=lambda item: item["directory"])
        return results, skipped

    futures_args = [str(csv_path) for csv_path in csv_paths]

    def run_with_executor(
        executor_cls: type[concurrent.futures.Executor],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
        results: List[Dict[str, object]] = []
        skipped: List[Dict[str, str]] = []
        with executor_cls(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    scan_single_csv_safe,
                    csv_path_str,
                    match_mode,
                    op_names,
                    loop_name,
                )
                for csv_path_str in futures_args
            ]
            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()
                if result is not None:
                    results.append(result)
                elif error is not None:
                    skipped.append(error)
        return results, skipped

    try:
        results, skipped = run_with_executor(concurrent.futures.ProcessPoolExecutor)
    except (OSError, PermissionError):
        results, skipped = run_with_executor(concurrent.futures.ThreadPoolExecutor)

    results.sort(key=lambda item: str(item["csv_path"]))
    skipped.sort(key=lambda item: item["directory"])
    return results, skipped


def append_record(
    records: List[Dict[str, object]],
    percentile_fields: Sequence[str],
    percentiles: Sequence[float],
    match_mode: str,
    scope: str,
    experiment: str,
    profile_name: str,
    rank_name: str,
    op_name: str,
    values: List[float],
    csv_count: int,
) -> None:
    stats = summarize(values, percentiles)
    record: Dict[str, object] = {
        "match_mode": match_mode,
        "scope": scope,
        "experiment": experiment,
        "profile_name": profile_name,
        "rank_name": rank_name,
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
    match_mode: str,
    op_names: Sequence[str],
    loop_name: str,
    workers: int,
    percentiles: Sequence[float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    records: List[Dict[str, object]] = []
    percentile_fields = [percentile_field_name(value) for value in percentiles]
    profile_samples: DefaultDict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    experiment_samples: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    overall_samples: DefaultDict[str, List[float]] = defaultdict(list)
    profile_csvs: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    experiment_csvs: DefaultDict[str, int] = defaultdict(int)
    csv_results, skipped = collect_csv_results(csv_paths, match_mode, op_names, loop_name, workers)

    for result in csv_results:
        experiment = str(result["experiment"])
        profile_name = str(result["profile_name"])
        rank_name = str(result["rank_name"])
        profile_csvs[(experiment, profile_name, rank_name)] += 1
        experiment_csvs[experiment] += 1
        matched = result["matched"]
        for op_name, durations in matched.items():
            if not durations:
                continue
            profile_samples[(experiment, profile_name, rank_name, op_name)].extend(durations)
            experiment_samples[(experiment, op_name)].extend(durations)
            overall_samples[op_name].extend(durations)

    for (experiment, profile_name, rank_name, op_name), values in sorted(profile_samples.items()):
        append_record(
            records=records,
            percentile_fields=percentile_fields,
            percentiles=percentiles,
            match_mode=match_mode,
            scope="profile",
            experiment=experiment,
            profile_name=profile_name,
            rank_name=rank_name,
            op_name=op_name,
            values=values,
            csv_count=profile_csvs[(experiment, profile_name, rank_name)],
        )

    for (experiment, op_name), values in sorted(experiment_samples.items()):
        append_record(
            records=records,
            percentile_fields=percentile_fields,
            percentiles=percentiles,
            match_mode=match_mode,
            scope="experiment",
            experiment=experiment,
            profile_name="ALL",
            rank_name="ALL",
            op_name=op_name,
            values=values,
            csv_count=experiment_csvs[experiment],
        )

    for op_name, values in sorted(overall_samples.items()):
        append_record(
            records=records,
            percentile_fields=percentile_fields,
            percentiles=percentiles,
            match_mode=match_mode,
            scope="overall",
            experiment="ALL",
            profile_name="ALL",
            rank_name="ALL",
            op_name=op_name,
            values=values,
            csv_count=len(csv_paths),
        )

    return records, skipped


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
            str(record.get("match_mode") or ""),
            str(record.get("op_name") or ""),
            scope_order.get(str(record.get("scope") or ""), 99),
            str(record.get("experiment") or ""),
            str(record.get("profile_name") or ""),
            str(record.get("rank_name") or ""),
        ),
    )


def write_csv(
    output_path: Path,
    records: List[Dict[str, object]],
    percentile_fields: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "match_mode",
        "scope",
        "experiment",
        "profile_name",
        "rank_name",
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
    op_names: Sequence[str],
    csv_paths: Sequence[Path],
    workers: int,
    match_mode: str,
    loop_name: str,
    percentiles: Sequence[float],
    records: Sequence[Dict[str, object]],
    skipped_csvs: Sequence[Dict[str, str]],
) -> str:
    percentile_fields = [percentile_field_name(value) for value in percentiles]
    lines = [
        "## Normal Kernel Summary",
        "",
        f"- 输入路径: `{input_path}`",
        f"- 命中 kernel_details.csv 数量: `{len(csv_paths)}`",
        f"- 成功解析数量: `{len(csv_paths) - len(skipped_csvs)}`",
        f"- 跳过数量: `{len(skipped_csvs)}`",
        f"- 并行进程数: `{workers}`",
        f"- 输出 CSV: `{output_path}`",
        f"- 匹配模式: `{match_mode}`",
        f"- 目标列表: `{', '.join(op_names)}`",
        f"- 分位数: `{', '.join(str(int(value)) if value.is_integer() else str(value) for value in percentiles)}`",
    ]
    if match_mode == "loop":
        lines.append(f"- 循环名称: `{loop_name}`")
    overall_records = [record for record in records if record["scope"] == "overall"]
    if overall_records:
        lines.append("")
        lines.append("### Overall")
        for record in overall_records:
            summary_parts = [
                f"{record['op_name']}: count={record['sample_count']}",
                f"mean={format_float(record['mean_us'])} us",
            ]
            for field in percentile_fields:
                summary_parts.append(f"{field[:-3]}={format_float(record.get(field))} us")
            lines.append("- " + ", ".join(summary_parts))
    if skipped_csvs:
        lines.append("")
        lines.append("### Skipped")
        for item in skipped_csvs[:10]:
            lines.append(f"- `{item['directory']}`: {item['error']}")
        if len(skipped_csvs) > 10:
            lines.append(f"- 其余跳过目录数: {len(skipped_csvs) - 10}")
    return "\n".join(lines)


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_normal_kernel_summary.csv")
    prefix = input_path.name or "benchmark_result"
    return Path.cwd() / f"{prefix}_normal_kernel_summary.csv"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "递归统计 normal 场景下给定算子在所有 kernel_details.csv 中的耗时分位数；"
            "支持单算子匹配，也支持按有序算子序列统计完整循环总耗时。"
        )
    )
    parser.add_argument("input_path", help="kernel_details.csv 路径，或包含多个实验结果的目录")
    parser.add_argument(
        "--ops",
        default="",
        help=(
            "目标算子列表，逗号分隔。"
            "在 op 模式下按名称规范化后做 exact/prefix 匹配；"
            "在 loop 模式下按给定顺序匹配完整循环。"
        ),
    )
    parser.add_argument(
        "--ops-file",
        help=(
            "从文件读取目标列表。"
            "支持每行一个算子，也支持每行逗号分隔多个算子；"
            "空行和以 # 开头的行会被忽略。"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("op", "loop"),
        default="op",
        help="统计模式：op=逐算子统计，loop=按有序算子序列统计每个完整循环的总耗时",
    )
    parser.add_argument(
        "--loop-name",
        default="loop_total",
        help="loop 模式下输出到 CSV 的循环名称，默认 loop_total",
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

    if args.ops and args.ops_file:
        print("请只使用 --ops 或 --ops-file 其中一种方式提供目标列表", file=sys.stderr)
        return 1

    op_names: List[str] = []
    if args.ops_file:
        ops_file_path = Path(args.ops_file).expanduser().resolve()
        if not ops_file_path.exists():
            print(f"--ops-file 指定的文件不存在: {ops_file_path}", file=sys.stderr)
            return 1
        op_names = parse_ops_file(ops_file_path)
    else:
        op_names = parse_op_list(args.ops)

    if not op_names:
        print("请通过 --ops 或 --ops-file 指定至少一个算子名", file=sys.stderr)
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
        print(f"未找到 kernel_details.csv: {input_path}", file=sys.stderr)
        return 1

    auto_workers = min(len(csv_paths), os.cpu_count() or 1)
    workers = args.workers if args.workers is not None else auto_workers
    workers = max(1, workers)

    all_records, skipped_csvs = build_records(
        csv_paths=csv_paths,
        match_mode=args.mode,
        op_names=op_names,
        loop_name=args.loop_name,
        workers=workers,
        percentiles=percentiles,
    )
    records = filter_records_by_scopes(all_records, scopes)
    if not all_records:
        if skipped_csvs:
            for item in skipped_csvs:
                print(f"- {item['directory']}: {item['error']}", file=sys.stderr)
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
                    "ops": op_names,
                    "ops_file": str(Path(args.ops_file).expanduser().resolve()) if args.ops_file else None,
                    "mode": args.mode,
                    "loop_name": args.loop_name,
                    "percentiles": percentiles,
                    "scopes": sorted(scopes),
                    "csv_count": len(csv_paths),
                    "workers": workers,
                    "skipped_csvs": skipped_csvs,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if skipped_csvs:
            for item in skipped_csvs:
                print(f"- {item['directory']}: {item['error']}", file=sys.stderr)
        print(
            render_markdown(
                input_path=input_path,
                output_path=output_path,
                op_names=op_names,
                csv_paths=csv_paths,
                workers=workers,
                match_mode=args.mode,
                loop_name=args.loop_name,
                percentiles=percentiles,
                records=records,
                skipped_csvs=skipped_csvs,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
