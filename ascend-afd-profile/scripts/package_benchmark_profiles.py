#!/usr/bin/env python3
"""Package benchmark.log and sampled profile files for benchmark experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


PROFILE_SUBDIRS = ("model_runner", "ffn")


@dataclass(frozen=True)
class Experiment:
    root: Path
    relative_dir: Path
    benchmark_log: Path

    @property
    def source_dir(self) -> Path:
        return self.benchmark_log.parent.parent


@dataclass
class PackageResult:
    experiment: Experiment
    copied_files: List[Path]
    warnings: List[str]
    archive_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect each experiment's log/benchmark.log and sampled files from "
            "profile/model_runner and profile/ffn, then create per-experiment "
            "archives and one final archive."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Benchmark result root, e.g. benchmark_results or benchmark_results/deepseek-v3.2.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input_dir>/profile_benchmark_package_<timestamp>.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=4,
        help="Number of files to copy from each profile subdirectory. Default: 4.",
    )
    parser.add_argument(
        "--archive-name",
        default=None,
        help="Final tar.gz file name. Defaults to <input_name>_profile_benchmark_<timestamp>.tar.gz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory first if it already exists.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {args.input_dir}")
    if args.sample_count < 0:
        raise SystemExit("--sample-count must be >= 0")


def discover_experiments(input_dir: Path) -> List[Experiment]:
    root = input_dir.resolve()
    experiments: List[Experiment] = []
    for benchmark_log in sorted(root.rglob("log/benchmark.log")):
        if not benchmark_log.is_file():
            continue
        experiment_dir = benchmark_log.parent.parent
        experiments.append(
            Experiment(
                root=root,
                relative_dir=experiment_dir.relative_to(root),
                benchmark_log=benchmark_log,
            )
        )
    return experiments


def iter_profile_files(profile_dir: Path) -> Iterable[Path]:
    if not profile_dir.is_dir():
        return []
    return sorted(path for path in profile_dir.rglob("*") if path.is_file())


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_archive_stem(relative_dir: Path) -> str:
    return "__".join(relative_dir.parts)


def make_tar_gz(
    source_path: Path,
    archive_path: Path,
    arcname: Path | str,
    exclude: Callable[[Path], bool] | None = None,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    def tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if exclude is None:
            return tarinfo
        member_path = source_path.parent / tarinfo.name
        if exclude(member_path.resolve()):
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_path, arcname=str(arcname), filter=tar_filter)


def package_experiment(
    experiment: Experiment,
    collected_root: Path,
    experiment_archives_dir: Path,
    sample_count: int,
) -> PackageResult:
    destination_dir = collected_root / experiment.relative_dir
    copied_files: List[Path] = []
    warnings: List[str] = []

    benchmark_destination = destination_dir / "log" / "benchmark.log"
    copy_file(experiment.benchmark_log, benchmark_destination)
    copied_files.append(benchmark_destination)

    for subdir in PROFILE_SUBDIRS:
        source_profile_dir = experiment.source_dir / "profile" / subdir
        if not source_profile_dir.is_dir():
            warnings.append(f"missing profile/{subdir}: {source_profile_dir}")
            continue

        selected_files = list(iter_profile_files(source_profile_dir))[:sample_count]
        if len(selected_files) < sample_count:
            warnings.append(
                f"profile/{subdir} has only {len(selected_files)} file(s): {source_profile_dir}"
            )

        for source_file in selected_files:
            relative_file = source_file.relative_to(source_profile_dir)
            destination_file = destination_dir / "profile" / subdir / relative_file
            copy_file(source_file, destination_file)
            copied_files.append(destination_file)

    archive_path = experiment_archives_dir / f"{safe_archive_stem(experiment.relative_dir)}.tar.gz"
    make_tar_gz(destination_dir, archive_path, experiment.relative_dir)

    return PackageResult(
        experiment=experiment,
        copied_files=copied_files,
        warnings=warnings,
        archive_path=archive_path,
    )


def write_manifest(output_dir: Path, results: Sequence[PackageResult]) -> Path:
    manifest_path = output_dir / "MANIFEST.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for result in results:
            manifest.write(f"[experiment] {result.experiment.relative_dir}\n")
            manifest.write(f"archive: {result.archive_path.relative_to(output_dir)}\n")
            for copied_file in sorted(result.copied_files):
                manifest.write(f"file: {copied_file.relative_to(output_dir)}\n")
            for warning in result.warnings:
                manifest.write(f"warning: {warning}\n")
            manifest.write("\n")
    return manifest_path


def prepare_output_dir(input_dir: Path, output_dir: Path | None, overwrite: bool) -> Path:
    if output_dir is None:
        output_dir = input_dir / f"profile_benchmark_package_{timestamp()}"
    output_dir = output_dir.resolve()

    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"output directory already exists, use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_dir = args.input_dir.resolve()
    output_dir = prepare_output_dir(input_dir, args.output_dir, args.overwrite)
    collected_root = output_dir / "collected"
    experiment_archives_dir = output_dir / "experiment_archives"

    experiments = discover_experiments(input_dir)
    if not experiments:
        raise SystemExit(f"no experiments found under: {input_dir}")

    results = [
        package_experiment(
            experiment=experiment,
            collected_root=collected_root,
            experiment_archives_dir=experiment_archives_dir,
            sample_count=args.sample_count,
        )
        for experiment in experiments
    ]

    manifest_path = write_manifest(output_dir, results)

    archive_name = args.archive_name
    if archive_name is None:
        archive_name = f"{input_dir.name}_profile_benchmark_{timestamp()}.tar.gz"
    final_archive_path = output_dir / archive_name
    make_tar_gz(
        output_dir,
        final_archive_path,
        output_dir.name,
        exclude=lambda path: path == final_archive_path.resolve(),
    )

    warning_count = sum(len(result.warnings) for result in results)
    copied_count = sum(len(result.copied_files) for result in results)
    print(f"experiments: {len(results)}")
    print(f"copied files: {copied_count}")
    print(f"warnings: {warning_count}")
    print(f"manifest: {manifest_path}")
    print(f"final archive: {final_archive_path}")
    if warning_count:
        print("warning details are written to MANIFEST.txt")


if __name__ == "__main__":
    main()
