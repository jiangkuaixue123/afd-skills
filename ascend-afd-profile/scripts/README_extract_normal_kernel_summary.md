# extract_normal_kernel_summary.py

用于统计 normal 场景下 `kernel_details.csv` 中给定算子列表的耗时分布。

脚本会递归扫描输入路径下所有名为 `kernel_details.csv` 的文件，匹配指定算子名，并输出这些算子的耗时统计结果，包括：

- `mean_us`
- `p25_us`
- `p50_us`
- `p75_us`
- `p90_us`
- `p99_us`

输出结果支持 3 个层级：

- `profile`：每个 profile 一行
- `experiment`：每个实验聚合一行
- `overall`：所有实验总体聚合一行

## 适用目录

适合类似下面这种 normal profile 目录：

```text
benchmark_results/
  deepseek-v3.2/
    normal_xxx/
      profile/
        jcz_afd_111/
          model_runner/
            xxx_ascend_pt/
              ASCEND_PROFILER_OUTPUT/
                kernel_details.csv
```

脚本会自动从路径中识别：

- `experiment`
- `rank_name`
- `profile_name`

## 匹配规则

通过 `--ops` 传入待统计的算子列表，多个算子用逗号分隔。

例如：

```bash
--ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

匹配时会先做名字归一化：

- 转小写
- 去掉非字母数字字符

然后按以下规则匹配：

- 完全匹配
- 或者 kernel 名以目标算子名为前缀

例如：

- `QuantBatchMatmulV3` 可以匹配 `QuantBatchMatmulV3_ND_NZ_int8_int8_bf16_high_performance_24`
- `SwiGlu` 可以匹配 `SwiGlu_3_high_performance_27`

## 基本用法

### 1. 统计一个 benchmark 根目录下所有实验

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

如果输入是目录，默认输出到当前工作目录：

```text
./deepseek-v3.2_normal_kernel_summary.csv
```

### 2. 只统计一个实验目录

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2/normal_bs24_dp32_tp1_in4096_20260409_143641 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

### 3. 只统计一个具体的 kernel_details.csv

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/kernel_details.csv \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

## 只输出某些层级

通过 `--scopes` 控制输出哪些层级，多个值用逗号分隔。

可选值：

- `profile`
- `experiment`
- `overall`

示例：

### 1. 只输出 overall

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --scopes overall
```

### 2. 只输出 experiment

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --scopes experiment
```

### 3. 同时输出 experiment 和 overall

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --scopes experiment,overall
```

## 指定输出文件

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --scopes overall \
  --output ./overall_only.csv
```

## 并行处理

当实验组很多、`kernel_details.csv` 很多时，可以通过 `--workers` 指定并行进程数：

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --scopes experiment \
  --workers 8
```

说明：

- 默认会自动选择并发数
- `--workers 1` 表示关闭并行
- 脚本优先使用多进程
- 如果当前环境不允许创建进程池，会自动回退到线程池

## 输出 CSV 字段说明

输出 CSV 包含以下列：

- `scope`：统计层级，取值为 `profile` / `experiment` / `overall`
- `experiment`：实验目录名
- `rank_name`：rank 目录名
- `profile_name`：profile 名
- `op_name`：目标算子名
- `csv_count`：该层级汇总了多少个 `kernel_details.csv`
- `sample_count`：该算子命中的 kernel 样本数
- `mean_us`
- `p25_us`
- `p50_us`
- `p75_us`
- `p90_us`
- `p99_us`

## CSV 排序规则

输出 CSV 按以下顺序排序：

1. `op_name`
2. `scope`，顺序为 `overall -> experiment -> profile`
3. `experiment`
4. `rank_name`
5. `profile_name`

也就是说，同一个算子的 overall / experiment / profile 结果会排在一起，方便对比。

## 终端输出

默认 `--format markdown`，终端会打印一个简短摘要。

如果指定：

```bash
--format json
```

会直接输出 JSON 结果，便于后续脚本处理。

## 实际示例

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /Users/jiangchenzhou/code/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/MY_code/deepseek-common/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --scopes overall
```

## 注意事项

- 输入路径既可以是目录，也可以是单个 `kernel_details.csv`
- 如果给定算子没有匹配到任何 kernel，脚本会报错退出
- 当前分位数统计直接基于全部命中样本，不做去极值处理
