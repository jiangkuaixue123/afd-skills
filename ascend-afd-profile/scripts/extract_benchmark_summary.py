#!/usr/bin/env python3
"""Summarize benchmark.log metrics for each experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PERF_ROW_RE = re.compile(
    r"^\s*│\s*(?P<metric>[^│]+?)\s*│\s*(?P<stage>[^│]+?)\s*│\s*(?P<avg>[^│]+?)\s*│\s*"
    r"(?P<min>[^│]+?)\s*│\s*(?P<max>[^│]+?)\s*│\s*(?P<median>[^│]+?)\s*│\s*(?P<p75>[^│]+?)\s*│\s*"
    r"(?P<p90>[^│]+?)\s*│\s*(?P<p99>[^│]+?)\s*│\s*(?P<count>[^│]+?)\s*│\s*$"
)
COMMON_ROW_RE = re.compile(
    r"^\s*│\s*(?P<metric>[^│]+?)\s*│\s*(?P<stage>[^│]+?)\s*│\s*(?P<value>[^│]+?)\s*│\s*$"
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def parse_float(text: str) -> Optional[float]:
    match = NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_int(text: str) -> Optional[int]:
    value = parse_float(text)
    if value is None:
        return None
    return int(round(value))


def discover_benchmark_logs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.name == "benchmark.log" else []
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("benchmark.log") if path.is_file())
    return []


def infer_experiment_name(log_path: Path) -> str:
    names: List[str] = []
    current = log_path.parent
    for _ in range(3):
        if not current.name:
            break
        names.append(current.name)
        current = current.parent
    names.reverse()
    return "/".join(names)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def collect_perf_rows(lines: Sequence[str]) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    for raw_line in lines:
        line = strip_ansi(raw_line.rstrip("\n"))
        match = PERF_ROW_RE.match(line)
        if not match:
            continue
        metric = match.group("metric").strip()
        rows[metric] = {
            "stage": match.group("stage").strip(),
            "avg": match.group("avg").strip(),
            "min": match.group("min").strip(),
            "max": match.group("max").strip(),
            "median": match.group("median").strip(),
            "p75": match.group("p75").strip(),
            "p90": match.group("p90").strip(),
            "p99": match.group("p99").strip(),
            "count": match.group("count").strip(),
        }
    return rows


def collect_common_rows(lines: Sequence[str]) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for raw_line in lines:
        line = strip_ansi(raw_line.rstrip("\n"))
        match = COMMON_ROW_RE.match(line)
        if not match:
            continue
        metric = match.group("metric").strip()
        rows[metric] = match.group("value").strip()
    return rows


def parse_benchmark_log(log_path: Path) -> Dict[str, object]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    perf_rows = collect_perf_rows(lines)
    common_rows = collect_common_rows(lines)

    tpot = perf_rows.get("TPOT")
    ott = perf_rows.get("OutputTokenThroughput")
    if tpot is None:
        raise ValueError(f"未在 {log_path} 中找到 TPOT 行")
    if ott is None:
        raise ValueError(f"未在 {log_path} 中找到 OutputTokenThroughput 行")

    record: Dict[str, object] = {
        "experiment": infer_experiment_name(log_path),
        "request_count": parse_int(tpot["count"]),
        "avg_concurrency": parse_float(common_rows.get("Concurrency", "")),
        "max_concurrency": parse_int(common_rows.get("Max Concurrency", "")),
        "tpot_mean_ms": parse_float(tpot["avg"]),
        "tpot_min_ms": parse_float(tpot["min"]),
        "tpot_max_ms": parse_float(tpot["max"]),
        "tpot_p50_ms": parse_float(tpot["median"]),
        "tpot_p75_ms": parse_float(tpot["p75"]),
        "tpot_p90_ms": parse_float(tpot["p90"]),
        "tpot_p99_ms": parse_float(tpot["p99"]),
        "output_token_throughput_global_token_s": parse_float(
            common_rows.get("Output Token Throughput", "")
        ),
    }
    return record


def parse_single_log(log_path_str: str) -> Dict[str, object]:
    return parse_benchmark_log(Path(log_path_str))


def parse_single_log_safe(log_path_str: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, str]]]:
    try:
        return parse_single_log(log_path_str), None
    except Exception as exc:
        return None, {"benchmark_log": log_path_str, "error": str(exc)}


def collect_records(
    log_paths: Sequence[Path],
    workers: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    if len(log_paths) <= 1 or workers <= 1:
        records: List[Dict[str, object]] = []
        skipped: List[Dict[str, str]] = []
        for log_path in log_paths:
            record, error = parse_single_log_safe(str(log_path))
            if record is not None:
                records.append(record)
            elif error is not None:
                skipped.append(error)
        records.sort(key=lambda item: str(item["experiment"]))
        skipped.sort(key=lambda item: item["benchmark_log"])
        return records, skipped

    def run_with_executor(
        executor_cls: type[concurrent.futures.Executor],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
        records: List[Dict[str, object]] = []
        skipped: List[Dict[str, str]] = []
        with executor_cls(max_workers=workers) as executor:
            futures = [
                executor.submit(parse_single_log_safe, str(log_path))
                for log_path in log_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                record, error = future.result()
                if record is not None:
                    records.append(record)
                elif error is not None:
                    skipped.append(error)
        return records, skipped

    try:
        records, skipped = run_with_executor(concurrent.futures.ProcessPoolExecutor)
    except (OSError, PermissionError):
        records, skipped = run_with_executor(concurrent.futures.ThreadPoolExecutor)

    records.sort(key=lambda item: str(item["experiment"]))
    skipped.sort(key=lambda item: item["benchmark_log"])
    return records, skipped


def write_csv(output_path: Path, records: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "request_count",
        "avg_concurrency",
        "max_concurrency",
        "tpot_mean_ms",
        "tpot_min_ms",
        "tpot_max_ms",
        "tpot_p50_ms",
        "tpot_p75_ms",
        "tpot_p90_ms",
        "tpot_p99_ms",
        "output_token_throughput_global_token_s",
    ]
    float_fields = {
        "avg_concurrency",
        "tpot_mean_ms",
        "tpot_min_ms",
        "tpot_max_ms",
        "tpot_p50_ms",
        "tpot_p75_ms",
        "tpot_p90_ms",
        "tpot_p99_ms",
        "output_token_throughput_global_token_s",
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: str(item["experiment"])):
            csv_record = dict(record)
            for field in float_fields:
                csv_record[field] = format_float(csv_record.get(field))
            writer.writerow(csv_record)


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_summary.csv")
    prefix = input_path.name or "benchmark_result"
    return Path.cwd() / f"{prefix}_benchmark_summary.csv"


def render_markdown(
    input_path: Path,
    output_path: Path,
    log_paths: Sequence[Path],
    workers: int,
    records: Sequence[Dict[str, object]],
    skipped_logs: Sequence[Dict[str, str]],
) -> str:
    lines = [
        "## Benchmark Summary",
        "",
        f"- 输入路径: `{input_path}`",
        f"- 命中 benchmark.log 数量: `{len(log_paths)}`",
        f"- 成功解析数量: `{len(records)}`",
        f"- 跳过数量: `{len(skipped_logs)}`",
        f"- 并行进程数: `{workers}`",
        f"- 输出 CSV: `{output_path}`",
    ]
    if records:
        lines.append("")
        lines.append("### Experiments")
        for record in records:
            lines.append(
                "- "
                f"{record['experiment']}: "
                f"TPOT mean={format_float(record.get('tpot_mean_ms'))} ms, "
                f"p90={format_float(record.get('tpot_p90_ms'))} ms, "
                f"avg_concurrency={format_float(record.get('avg_concurrency'))}, "
                f"max_concurrency={record.get('max_concurrency')}, "
                f"output_token_throughput={format_float(record.get('output_token_throughput_global_token_s'))} token/s"
            )
    if skipped_logs:
        lines.append("")
        lines.append("### Skipped")
        for item in skipped_logs[:10]:
            lines.append(f"- `{item['benchmark_log']}`: {item['error']}")
        if len(skipped_logs) > 10:
            lines.append(f"- 其余跳过文件数: {len(skipped_logs) - 10}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归统计 benchmark.log 中的 TPOT、并发和 Output Token Throughput。"
    )
    parser.add_argument("input_path", help="benchmark.log 路径，或包含多个实验目录的目录")
    parser.add_argument("-o", "--output", help="输出 CSV 路径；未指定时自动生成")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="终端输出格式，默认 markdown",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数；未指定时自动选择，传 1 可关闭并行",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}", file=sys.stderr)
        return 1

    log_paths = discover_benchmark_logs(input_path)
    if not log_paths:
        print(f"未找到 benchmark.log: {input_path}", file=sys.stderr)
        return 1

    auto_workers = min(len(log_paths), os.cpu_count() or 1)
    workers = args.workers if args.workers is not None else auto_workers
    workers = max(1, workers)

    records, skipped_logs = collect_records(log_paths, workers)
    if not records:
        print("没有成功解析任何 benchmark.log", file=sys.stderr)
        if skipped_logs:
            for item in skipped_logs[:10]:
                print(f"- {item['benchmark_log']}: {item['error']}", file=sys.stderr)
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
                    "workers": workers,
                    "log_count": len(log_paths),
                    "skipped_logs": skipped_logs,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_markdown(input_path, output_path, log_paths, workers, records, skipped_logs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
