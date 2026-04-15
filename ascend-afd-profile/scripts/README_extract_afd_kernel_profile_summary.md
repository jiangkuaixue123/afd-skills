# `extract_afd_kernel_profile_summary.py`

用于统计 AFD 场景下各实验目录中两侧 profile 的算子耗时分布，并把 Attention / FFN 结果合并到同一张 CSV。

脚本会递归扫描输入路径下所有名为 `kernel_details.csv` 的文件，但只保留位于以下目录分支中的 profile：

- `model_runner`：记为 `role=attn`
- `ffn`：记为 `role=ffn`

然后按传入的 Attention / FFN 算子列表分别匹配 kernel，并输出：

- `mean_us`
- `p25_us`
- `p50_us`
- `p75_us`
- `p90_us`
- `p99_us`

默认同时输出 3 个层级：

- `profile`：每个 profile/rank 一行
- `experiment`：每个实验、每个 role、每个算子聚合一行
- `overall`：全局每个 role、每个算子聚合一行

## 适用目录

适合类似下面这种 AFD profile 目录：

```text
benchmark_results/
  deepseek-v3.2/
    afd_xxx/
      profile/
        jcz_afd_111/
          model_runner/
            rank_0/
              ASCEND_PROFILER_OUTPUT/
                kernel_details.csv
          ffn/
            rank_8/
              ASCEND_PROFILER_OUTPUT/
                kernel_details.csv
```

脚本会自动从路径中识别：

- `experiment`
- `profile_name`
- `rank_name`
- `role`

其中：

- `model_runner -> attn`
- `ffn -> ffn`

支持两种常见目录：

```text
.../experiment/profile/<profile_name>/model_runner/<rank>/.../kernel_details.csv
.../experiment/profile/<profile_name>/ffn/<rank>/.../kernel_details.csv
```

以及扁平目录：

```text
.../experiment/profile/model_runner/<rank>/.../kernel_details.csv
.../experiment/profile/ffn/<rank>/.../kernel_details.csv
```

如果是扁平目录，没有额外的 `<profile_name>` 层，输出里的 `profile_name` 会写成 `DEFAULT_PROFILE`。

## 匹配规则

- `--attn-ops` 只用于匹配 `model_runner` 下的 `kernel_details.csv`
- `--ffn-ops` 只用于匹配 `ffn` 下的 `kernel_details.csv`

匹配时会先做名字归一化：

- 转小写
- 去掉非字母数字字符

然后按以下规则匹配：

- 完全匹配
- 或者 kernel 名以目标算子名为前缀

例如：

- `FusedInferAttentionScore` 可以匹配 `FusedInferAttentionScore_xxx`
- `GroupedMatmul` 可以匹配 `GroupedMatmul_xxx`

## 基本用法

### 1. 统计一个 benchmark 根目录下所有实验

```bash
python3 ascend-afd-profile/scripts/extract_afd_kernel_profile_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --attn-ops 'FusedInferAttentionScore,A2e,E2a' \
  --ffn-ops 'GroupedMatmul,A2e,E2a'
```

如果输入是目录，默认输出到当前工作目录：

```text
./deepseek-v3.2_afd_kernel_profile_summary.csv
```

### 2. 只统计一个实验目录

```bash
python3 ascend-afd-profile/scripts/extract_afd_kernel_profile_summary.py \
  /path/to/benchmark_results/deepseek-v3.2/afd_xxx \
  --attn-ops 'FusedInferAttentionScore,A2e,E2a' \
  --ffn-ops 'GroupedMatmul,A2e,E2a'
```

### 3. 自定义分位数

```bash
python3 ascend-afd-profile/scripts/extract_afd_kernel_profile_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --attn-ops 'FusedInferAttentionScore' \
  --ffn-ops 'GroupedMatmul' \
  --percentiles '50,80,90,95,99'
```

## 只输出某些层级

通过 `--scopes` 控制输出哪些层级，多个值用逗号分隔。

可选值：

- `profile`
- `experiment`
- `overall`

例如只输出实验聚合：

```bash
python3 ascend-afd-profile/scripts/extract_afd_kernel_profile_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --attn-ops 'FusedInferAttentionScore' \
  --ffn-ops 'GroupedMatmul' \
  --scopes experiment
```

## 输出 CSV 字段说明

- `scope`：统计层级，取值为 `profile` / `experiment` / `overall`
- `experiment`：实验目录名
- `profile_name`：profile 名
- `rank_name`：rank 目录名
- `role`：`attn` 或 `ffn`
- `op_name`：目标算子名
- `csv_count`：该层级汇总了多少个 `kernel_details.csv`
- `sample_count`：该算子命中的 kernel 样本数
- `mean_us`
- `pXX_us`：由 `--percentiles` 决定，默认是 `p25_us/p50_us/p75_us/p90_us/p99_us`

## CSV 排序规则

输出 CSV 按以下顺序排序：

1. `role`
2. `op_name`
3. `scope`，顺序为 `overall -> experiment -> profile`
4. `experiment`
5. `profile_name`
6. `rank_name`

## 终端输出

默认 `--format markdown`，终端会打印 overall 摘要。

如果指定：

```bash
--format json
```

会直接输出 JSON 结果，便于后续脚本处理。

## 注意事项

- 输入路径既可以是目录，也可以是单个 `kernel_details.csv`
- 只有路径中包含 `model_runner` 或 `ffn` 的 `kernel_details.csv` 会被统计
- 至少需要指定 `--attn-ops` 或 `--ffn-ops` 其中之一
