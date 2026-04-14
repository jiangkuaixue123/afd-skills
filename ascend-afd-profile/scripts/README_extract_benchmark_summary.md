# extract_benchmark_summary.py

用于递归统计 `benchmark.log` 中的 benchmark summary 信息，按实验输出一行。

当前会抽取以下指标：

- `TPOT` 的 `mean / min / max / p50 / p75 / p90 / p99`
- `Concurrency`
- `Max Concurrency`
- `Output Token Throughput` 的全局吞吐值

说明：

- `TPOT` 来自 benchmark 结果表
- `Concurrency`、`Max Concurrency`、`Output Token Throughput` 来自 common metrics 表
- 每个实验目录下默认只有一个 `benchmark.log`

## 适用目录

适合类似下面这种结构：

```text
benchmark_results/
  deepseek-v3.2/
    normal_xxx/
      benchmark.log
      profile/
        ...
```

脚本会递归查找输入路径下所有 `benchmark.log`。

## 基本用法

### 1. 统计一个 benchmark 根目录下所有实验

```bash
python3 ascend-afd-profile/scripts/extract_benchmark_summary.py \
  /path/to/benchmark_results/deepseek-v3.2
```

### 2. 只统计一个实验目录

```bash
python3 ascend-afd-profile/scripts/extract_benchmark_summary.py \
  /path/to/benchmark_results/deepseek-v3.2/normal_bs24_dp32_tp1_in4096_20260409_143641
```

### 3. 只统计单个 benchmark.log

```bash
python3 ascend-afd-profile/scripts/extract_benchmark_summary.py \
  /path/to/benchmark.log
```

## 并行处理

实验很多时可以开启并行：

```bash
python3 ascend-afd-profile/scripts/extract_benchmark_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --workers 8
```

说明：

- 默认自动选择并发数
- `--workers 1` 表示关闭并行
- 脚本优先使用多进程
- 如果当前环境不允许创建进程池，会自动回退到线程池

## 输出字段

CSV 输出包含以下列：

- `experiment`
- `request_count`
- `avg_concurrency`
- `max_concurrency`
- `tpot_mean_ms`
- `tpot_min_ms`
- `tpot_max_ms`
- `tpot_p50_ms`
- `tpot_p75_ms`
- `tpot_p90_ms`
- `tpot_p99_ms`
- `output_token_throughput_global_token_s`

其中：

- 所有浮点字段保留 3 位小数
- `experiment` 默认取 `benchmark.log` 所在目录名

## 示例

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_benchmark_summary.py \
  /Users/jiangchenzhou/code/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/MY_code/deepseek-common/benchmark_results/deepseek-v3.2
```
