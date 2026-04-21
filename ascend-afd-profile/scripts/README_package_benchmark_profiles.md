# package_benchmark_profiles.py

将 benchmark 目录下每个实验的 `log/benchmark.log`，以及 `profile/model_runner` 和 `profile/ffn` 中各指定数量的 profile 目录取出，按实验目录整理，并生成：

- `collected/`: 按实验名放置的文件目录
- `experiment_archives/`: 每个实验一个 `.tar.gz`
- `<input_name>_profile_benchmark_<timestamp>.tar.gz`: 总压缩包
- `MANIFEST.txt`: 复制文件和缺失目录记录

## 适用目录

支持输入模型目录：

```text
benchmark_results/deepseek-v3.2/
  normal_xxx/
    log/benchmark.log
    profile/model_runner/
    profile/ffn/
```

也支持输入更上层的 `benchmark_results/`，脚本会保留相对路径，例如 `deepseek-v3.2/normal_xxx`。

## 用法

```bash
python3 ascend-afd-profile/scripts/package_benchmark_profiles.py \
  /path/to/benchmark_results/deepseek-v3.2
```

指定输出目录：

```bash
python3 ascend-afd-profile/scripts/package_benchmark_profiles.py \
  /path/to/benchmark_results \
  -o /path/to/output/package_profiles \
  --overwrite
```

调整每侧抽取的 profile 目录数量：

```bash
python3 ascend-afd-profile/scripts/package_benchmark_profiles.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --profile-count 2
```

调整每个 profile 目录内抽取的文件数量：

```bash
python3 ascend-afd-profile/scripts/package_benchmark_profiles.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --sample-count 4
```

脚本默认会在终端显示进度条，包括实验打包进度、manifest 写入和总压缩包创建进度。关闭进度显示：

```bash
python3 ascend-afd-profile/scripts/package_benchmark_profiles.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --no-progress
```

性能相关参数：

```bash
python3 ascend-afd-profile/scripts/package_benchmark_profiles.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --profile-count 2 \
  --sample-count 4 \
  --workers 8 \
  --compression-level 1 \
  --no-progress
```

- `--workers`: 每个实验独立打包时的并行 worker 数，默认自动选择，最多 8 个。
- `--compression-level`: gzip 压缩等级，范围 0-9。默认 1，优先打包速度；如果更关注包体大小可以调高。
- `--no-progress`: 关闭逐文件进度显示，并跳过最终压缩包创建前的文件数统计扫描。

## 输出示例

```text
profile_benchmark_package_20260421_120000/
  MANIFEST.txt
  collected/
    afd_bs24_32A16F_ub2_in4096_e4_20260421_002911/
      log/benchmark.log
      profile/
        model_runner/<最多1个profile目录，每个目录最多4个文件>
        ffn/<最多1个profile目录，每个目录最多4个文件>
  experiment_archives/
    afd_bs24_32A16F_ub2_in4096_e4_20260421_002911.tar.gz
  deepseek-v3.2_profile_benchmark_20260421_120000.tar.gz
```
