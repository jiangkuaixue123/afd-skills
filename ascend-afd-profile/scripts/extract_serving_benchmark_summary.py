#!/usr/bin/env python3
"""Summarize serving benchmark.log metrics for each experiment."""

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
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
RESULT_HEADER = "============ Serving Benchmark Result ============"
SECTION_LINES = {
    "---------------Time to First Token----------------": "ttft",
    "-----Time per Output Token (excl. 1st token)------": "tpot",
}
EXCLUDED_FIELDS = {
    "benchmark_duration_s",
    "failed_requests",
    "total_generated_tokens",
    "total_input_tokens",
    "total_token_throughput_tok_s",
}
CSV_FIELD_ORDER = [
    "experiment",
    "successful_requests",
    "maximum_request_concurrency",
    "peak_concurrent_requests",
    "request_throughput_req_s",
    "output_token_throughput_tok_s",
    "peak_output_token_throughput_tok_s",
    "tpot_mean_ms",
    "tpot_median_ms",
    "tpot_p25_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p95_ms",
    "tpot_p99_ms",
    "single_card_output_token_throughput_tok_s",
    "single_card_peak_output_token_throughput_tok_s",
]


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


def normalize_metric_key(text: str) -> str:
    stripped = text.lower().strip()
    unit_parts = re.findall(r"\(([^)]*)\)", stripped)
    base = re.sub(r"\([^)]*\)", "", stripped)
    parts = [base, *unit_parts]
    sanitized = "_".join(
        re.sub(r"[^a-z0-9]+", "_", part).strip("_")
        for part in parts
        if part.strip()
    )
    return sanitized.strip("_")


def parse_result_block(lines: Sequence[str]) -> Dict[str, object]:
    result_start = None
    for index, line in enumerate(lines):
        if line.strip() == RESULT_HEADER:
            result_start = index + 1
            break
    if result_start is None:
        raise ValueError("未找到 Serving Benchmark Result 段")

    record: Dict[str, object] = {}
    current_prefix = ""
    for raw_line in lines[result_start:]:
        line = strip_ansi(raw_line).strip()
        if not line:
            continue
        if line.startswith("==="):
            break
        prefix = SECTION_LINES.get(line)
        if prefix is not None:
            current_prefix = prefix
            continue
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        base_key = normalize_metric_key(label)
        if current_prefix:
            base_key = re.sub(
                rf"(^|_){re.escape(current_prefix)}(_|$)",
                "_",
                base_key,
            ).strip("_")
        key = f"{current_prefix}_{base_key}" if current_prefix else base_key
        number = parse_float(value)
        if number is None:
            continue
        if number.is_integer():
            record[key] = int(number)
        else:
            record[key] = number
    if not record:
        raise ValueError("Serving Benchmark Result 段中未解析出任何指标")
    return record


def parse_benchmark_log(log_path: Path) -> Dict[str, object]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    record = parse_result_block(lines)
    record["experiment"] = infer_experiment_name(log_path)
    enrich_single_card_throughput(record)
    return record


def infer_card_count(experiment: str) -> Optional[int]:
    afd_match = re.search(r"(?:^|/)afd_[^/]*?_(\d+)A(\d+)F(?:_|/|$)", experiment, re.IGNORECASE)
    if afd_match:
        return int(afd_match.group(1)) + int(afd_match.group(2))

    normal_match = re.search(r"(?:^|/)normal_[^/]*?_dp(\d+)(?:_|/|$)", experiment, re.IGNORECASE)
    if normal_match:
        return int(normal_match.group(1))

    return None


def enrich_single_card_throughput(record: Dict[str, object]) -> None:
    experiment = str(record.get("experiment", ""))
    card_count = infer_card_count(experiment)
    if not card_count:
        return

    output_token_throughput = get_record_float(record, "output_token_throughput_tok_s")
    peak_output_token_throughput = get_record_float(record, "peak_output_token_throughput_tok_s")

    if output_token_throughput is not None:
        record["single_card_output_token_throughput_tok_s"] = (
            output_token_throughput / card_count
        )
    if peak_output_token_throughput is not None:
        record["single_card_peak_output_token_throughput_tok_s"] = (
            peak_output_token_throughput / card_count
        )
    record["card_count"] = card_count


def filter_record_for_output(record: Dict[str, object]) -> Dict[str, object]:
    filtered: Dict[str, object] = {}
    for key, value in record.items():
        if key == "experiment":
            filtered[key] = value
            continue
        if key.startswith("ttft_"):
            continue
        if key in EXCLUDED_FIELDS or key == "card_count":
            continue
        filtered[key] = value
    filtered.setdefault("single_card_output_token_throughput_tok_s", "")
    filtered.setdefault("single_card_peak_output_token_throughput_tok_s", "")
    return filtered


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


def fieldnames_from_records(records: Sequence[Dict[str, object]]) -> List[str]:
    available_keys = {
        key
        for record in records
        for key in record.keys()
        if key != "experiment"
    }
    ordered_keys = [field for field in CSV_FIELD_ORDER if field != "experiment"]
    remaining_keys = sorted(available_keys - set(ordered_keys))
    return ["experiment", *ordered_keys, *remaining_keys]


def write_csv(output_path: Path, records: Sequence[Dict[str, object]]) -> List[str]:
    fieldnames = fieldnames_from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: str(item["experiment"])):
            csv_record: Dict[str, object] = {}
            for field in fieldnames:
                value = record.get(field, "")
                if isinstance(value, float):
                    csv_record[field] = format_float(value)
                else:
                    csv_record[field] = value
            writer.writerow(csv_record)
    return fieldnames


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_serving_summary.csv")
    prefix = input_path.name or "benchmark_result"
    return Path.cwd() / f"{prefix}_serving_benchmark_summary.csv"


def get_record_float(record: Dict[str, object], key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_float(str(value))


def render_markdown(
    input_path: Path,
    output_path: Path,
    log_paths: Sequence[Path],
    workers: int,
    fieldnames: Sequence[str],
    records: Sequence[Dict[str, object]],
    skipped_logs: Sequence[Dict[str, str]],
) -> str:
    lines = [
        "## Serving Benchmark Summary",
        "",
        f"- 输入路径: `{input_path}`",
        f"- 命中 benchmark.log 数量: `{len(log_paths)}`",
        f"- 成功解析数量: `{len(records)}`",
        f"- 跳过数量: `{len(skipped_logs)}`",
        f"- 并行进程数: `{workers}`",
        f"- 输出 CSV: `{output_path}`",
        f"- 指标列数: `{max(0, len(fieldnames) - 1)}`",
    ]
    if records:
        lines.append("")
        lines.append("### Experiments")
        for record in records:
            lines.append(
                "- "
                f"{record['experiment']}: "
                f"request_throughput_req_s={format_float(get_record_float(record, 'request_throughput_req_s'))}, "
                f"output_token_throughput_tok_s={format_float(get_record_float(record, 'output_token_throughput_tok_s'))}, "
                f"single_card_output_token_throughput_tok_s="
                f"{format_float(get_record_float(record, 'single_card_output_token_throughput_tok_s'))}, "
                f"single_card_peak_output_token_throughput_tok_s="
                f"{format_float(get_record_float(record, 'single_card_peak_output_token_throughput_tok_s'))}, "
                f"tpot_mean_ms={format_float(get_record_float(record, 'tpot_mean_ms'))}"
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
        description="递归统计 serving benchmark.log 中的吞吐、TTFT、TPOT 等指标。"
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

    output_records = [filter_record_for_output(record) for record in records]

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )
    fieldnames = write_csv(output_path, output_records)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "workers": workers,
                    "log_count": len(log_paths),
                    "fieldnames": fieldnames,
                    "skipped_logs": skipped_logs,
                    "records": output_records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            render_markdown(
                input_path,
                output_path,
                log_paths,
                workers,
                fieldnames,
                output_records,
                skipped_logs,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
