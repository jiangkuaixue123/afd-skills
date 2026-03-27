#!/usr/bin/env python3
"""Scan benchmark_result experiments and summarize key AFD op latencies."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_./-]+)\s*=\s*([^\s,;]+)")

ATTENTION_KEYWORDS = (
    "attention",
    "flashattention",
    "flash_attn",
    "pagedattention",
    "paged_attention",
)


@dataclass
class OpStats:
    mean_samples: List[float] = field(default_factory=list)
    min_samples: List[float] = field(default_factory=list)
    max_samples: List[float] = field(default_factory=list)

    def add_mean(self, value: float) -> None:
        self.mean_samples.append(value)

    def add_min(self, value: float) -> None:
        self.min_samples.append(value)

    def add_max(self, value: float) -> None:
        self.max_samples.append(value)

    @property
    def count(self) -> int:
        return max(len(self.mean_samples), len(self.min_samples), len(self.max_samples))

    @staticmethod
    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def mean(self) -> Optional[float]:
        return self._avg(self.mean_samples)

    @property
    def min(self) -> Optional[float]:
        if not self.min_samples:
            return None
        return min(self.min_samples)

    @property
    def max(self) -> Optional[float]:
        if not self.max_samples:
            return None
        return max(self.max_samples)

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
        }


@dataclass
class SideSummary:
    side: str
    files: List[str] = field(default_factory=list)
    stats: Dict[str, OpStats] = field(default_factory=dict)

    def ensure_op(self, op_name: str) -> OpStats:
        if op_name not in self.stats:
            self.stats[op_name] = OpStats()
        return self.stats[op_name]


@dataclass
class ExperimentSummary:
    name: str
    path: str
    run_params_path: Optional[str]
    run_params: Dict[str, str]
    run_params_raw: List[str]
    attention: SideSummary
    ffn: SideSummary
    notes: List[str] = field(default_factory=list)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 benchmark_result 下各实验的 AFD profile 关键算子统计。"
    )
    parser.add_argument("benchmark_result", help="benchmark_result 根目录")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="输出格式，默认 markdown",
    )
    return parser.parse_args(argv)


def parse_run_params(path: Path) -> Tuple[Dict[str, str], List[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    scenario_started = False
    parsed: Dict[str, str] = {}
    raw: List[str] = []
    for line in lines:
        if not scenario_started:
            if "Test Scenario:" in line:
                scenario_started = True
                remainder = line.split("Test Scenario:", 1)[1].strip()
                if remainder:
                    raw.append(remainder)
                    for key, value in KEY_VALUE_RE.findall(remainder):
                        parsed[key] = value
            continue

        stripped = line.strip()
        if not stripped:
            continue
        raw.append(stripped)
        for key, value in KEY_VALUE_RE.findall(stripped):
            parsed[key] = value
    return parsed, raw


def sniff_headers(fieldnames: Iterable[str]) -> Dict[str, Optional[str]]:
    mapping = {
        "op_name": None,
        "avg": None,
        "min": None,
        "max": None,
        "total": None,
        "count": None,
    }
    for field in fieldnames:
        key = normalize(field)
        if not mapping["op_name"] and (
            key in {"opname", "operatorname", "name", "kernelname", "optype"}
            or key.startswith("opname")
            or key.startswith("operatorname")
        ):
            mapping["op_name"] = field
        if not mapping["avg"] and (
            key in {"avgtime", "avg", "averagetime", "mean", "meantime", "taskavg"}
            or "avgtime" in key
            or "averagetime" in key
            or key.startswith("mean")
        ):
            mapping["avg"] = field
        if not mapping["min"] and (
            key in {"mintime", "min", "minimumtime", "taskmin"}
            or "mintime" in key
            or "minimumtime" in key
        ):
            mapping["min"] = field
        if not mapping["max"] and (
            key in {"maxtime", "max", "maximumtime", "taskmax"}
            or "maxtime" in key
            or "maximumtime" in key
        ):
            mapping["max"] = field
        if not mapping["total"] and (
            key in {"totaltime", "total", "sumtime", "taskduration"}
            or "totaltime" in key
            or "sumtime" in key
            or "taskduration" in key
        ):
            mapping["total"] = field
        if not mapping["count"] and (
            key in {"count", "calls", "callcount", "occurrences"}
            or key.endswith("count")
            or "callcount" in key
        ):
            mapping["count"] = field
    return mapping


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


def parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return int(float(stripped))
    except ValueError:
        return None


def read_op_statistics(csv_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}
        headers = sniff_headers(reader.fieldnames)
        op_field = headers["op_name"]
        if not op_field:
            return {}

        result: Dict[str, Dict[str, Optional[float]]] = {}
        for row in reader:
            op_name = (row.get(op_field) or "").strip()
            if not op_name:
                continue

            avg = parse_float(row.get(headers["avg"])) if headers["avg"] else None
            min_value = parse_float(row.get(headers["min"])) if headers["min"] else None
            max_value = parse_float(row.get(headers["max"])) if headers["max"] else None
            total = parse_float(row.get(headers["total"])) if headers["total"] else None
            count = parse_int(row.get(headers["count"])) if headers["count"] else None

            if avg is None and total is not None and count and count > 0:
                avg = total / count

            result[op_name] = {
                "mean": avg,
                "min": min_value,
                "max": max_value,
                "count": count,
            }
        return result


def select_ops(side: str, op_rows: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    selected: Dict[str, Dict[str, Optional[float]]] = {}
    for required in ("A2e", "E2a"):
        for op_name, stats in op_rows.items():
            if normalize(op_name) == normalize(required):
                selected[op_name] = stats

    if side == "ffn":
        for required in ("GroupMatmul", "MoeDispatch", "MoeCombine"):
            for op_name, stats in op_rows.items():
                if normalize(op_name) == normalize(required):
                    selected[op_name] = stats

    if side == "attention":
        for op_name, stats in op_rows.items():
            lowered = op_name.lower()
            if any(keyword in lowered for keyword in ATTENTION_KEYWORDS):
                selected[op_name] = stats

    return selected


def summarize_side(side_dir: Path, side: str) -> SideSummary:
    summary = SideSummary(side=side)
    if not side_dir.exists():
        return summary

    csv_files = sorted(side_dir.rglob("op_statistic.csv"))
    summary.files = [str(path) for path in csv_files]
    for csv_file in csv_files:
        op_rows = read_op_statistics(csv_file)
        selected = select_ops(side, op_rows)
        for op_name, stats in selected.items():
            entry = summary.ensure_op(op_name)
            if stats.get("mean") is not None:
                entry.add_mean(stats["mean"])
            if stats.get("min") is not None:
                entry.add_min(stats["min"])
            if stats.get("max") is not None:
                entry.add_max(stats["max"])
    return summary


def summarize_experiment(exp_dir: Path) -> ExperimentSummary:
    run_params_path = exp_dir / "scripts" / "run_params.txt"
    run_params: Dict[str, str] = {}
    run_params_raw: List[str] = []
    notes: List[str] = []
    if run_params_path.exists():
        run_params, run_params_raw = parse_run_params(run_params_path)
        if not run_params:
            notes.append("未从 Test Scenario 段落解析出 key=value 配置。")
    else:
        notes.append("缺少 scripts/run_params.txt。")

    attention = summarize_side(exp_dir / "profile" / "attention", "attention")
    ffn = summarize_side(exp_dir / "profile" / "ffn", "ffn")
    if not attention.files:
        notes.append("Attention 侧未找到 op_statistic.csv。")
    if not ffn.files:
        notes.append("FFN 侧未找到 op_statistic.csv。")

    return ExperimentSummary(
        name=exp_dir.name,
        path=str(exp_dir),
        run_params_path=str(run_params_path) if run_params_path.exists() else None,
        run_params=run_params,
        run_params_raw=run_params_raw,
        attention=attention,
        ffn=ffn,
        notes=notes,
    )


def infer_bottleneck(exp: ExperimentSummary) -> List[str]:
    hints: List[str] = []
    for op in ("A2e", "E2a"):
        attention_stats = find_op_stats(exp.attention, op)
        ffn_stats = find_op_stats(exp.ffn, op)
        if not attention_stats or not ffn_stats:
            continue
        att_mean = attention_stats.mean
        ffn_mean = ffn_stats.mean
        if att_mean is None or ffn_mean is None:
            continue
        if att_mean > ffn_mean:
            hints.append(f"Attention 侧 {op} 更长，按经验规则优先怀疑 FFN 侧是当前时延瓶颈。")
        elif ffn_mean > att_mean:
            hints.append(f"FFN 侧 {op} 更长，按经验规则优先怀疑 Attention 侧是当前时延瓶颈。")

    for side_name, summary in (("Attention", exp.attention), ("FFN", exp.ffn)):
        for op_name, stats in summary.stats.items():
            if stats.mean is None or stats.max is None:
                continue
            if stats.max >= stats.mean * 1.5:
                hints.append(
                    f"{side_name} 侧 {op_name} 最大值明显高于均值，可能存在抖动或长尾。"
                )
    return hints


def find_op_stats(summary: SideSummary, required_name: str) -> Optional[OpStats]:
    target = normalize(required_name)
    for op_name, stats in summary.stats.items():
        if normalize(op_name) == target:
            return stats
    return None


def collect_experiments(root: Path) -> List[ExperimentSummary]:
    experiments: List[ExperimentSummary] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            experiments.append(summarize_experiment(child))
    return experiments


def render_side_markdown(title: str, summary: SideSummary) -> List[str]:
    lines = [f"- `{title}`"]
    if not summary.files:
        lines.append("  未找到 `op_statistic.csv`。")
        return lines

    lines.append(f"  找到 {len(summary.files)} 个 `op_statistic.csv`。")
    if not summary.stats:
        lines.append("  未匹配到目标算子。")
        return lines

    for op_name in sorted(summary.stats):
        stats = summary.stats[op_name]
        lines.append(
            "  "
            + f"{op_name}: mean={format_float(stats.mean)}, min={format_float(stats.min)}, max={format_float(stats.max)}"
        )
    return lines


def render_markdown(experiments: Sequence[ExperimentSummary]) -> str:
    lines = ["## 实验扫描结果", ""]
    lines.append(f"- 共发现 {len(experiments)} 组实验")
    for exp in experiments:
        lines.append(
            f"- {exp.name}: run_params={'Y' if exp.run_params_path else 'N'}, "
            f"attention_csv={len(exp.attention.files)}, ffn_csv={len(exp.ffn.files)}"
        )

    for exp in experiments:
        lines.extend(["", f"## {exp.name}", ""])
        lines.append(f"- 路径: `{exp.path}`")
        if exp.run_params:
            config = ", ".join(f"{key}={value}" for key, value in sorted(exp.run_params.items()))
            lines.append(f"- 配置摘要: {config}")
        elif exp.run_params_raw:
            lines.append(f"- 配置摘要: 未提取到 key=value，原始段落 {exp.run_params_raw}")
        else:
            lines.append("- 配置摘要: 未找到可解析配置")

        lines.extend(render_side_markdown("Attention 侧观察", exp.attention))
        lines.extend(render_side_markdown("FFN 侧观察", exp.ffn))

        hints = infer_bottleneck(exp)
        if hints:
            lines.append("- 综合判断:")
            for hint in hints:
                lines.append(f"  {hint}")

        if exp.notes:
            lines.append("- 说明:")
            for note in exp.notes:
                lines.append(f"  {note}")

    return "\n".join(lines)


def render_json(experiments: Sequence[ExperimentSummary]) -> str:
    payload = []
    for exp in experiments:
        payload.append(
            {
                "name": exp.name,
                "path": exp.path,
                "run_params_path": exp.run_params_path,
                "run_params": exp.run_params,
                "run_params_raw": exp.run_params_raw,
                "attention": {
                    "files": exp.attention.files,
                    "stats": {name: stats.as_dict() for name, stats in exp.attention.stats.items()},
                },
                "ffn": {
                    "files": exp.ffn.files,
                    "stats": {name: stats.as_dict() for name, stats in exp.ffn.stats.items()},
                },
                "notes": exp.notes,
                "hints": infer_bottleneck(exp),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = Path(args.benchmark_result).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"benchmark_result 目录不存在: {root}", file=sys.stderr)
        return 1

    experiments = collect_experiments(root)
    if args.format == "json":
        print(render_json(experiments))
    else:
        print(render_markdown(experiments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
