# `extract_kernel_stage_summary.py`

用于从 Ascend profile 的 `kernel_details.csv` 中统计去除通信 marker 后的阶段执行时间，并输出 CSV 汇总表。

## 脚本路径

```bash
/Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py
```

## 适用目录结构

### 1. `benchmark_result` 根目录模式

目录名可以任意，结构类似：

```text
benchmark_result
|____dir1
     |____log
     |____profile
     |    |____attention
     |    |____ffn
     |____scripts
          |____run_params.txt
|____dir2
     ...
```

在这种模式下，脚本会遍历所有一级实验目录，例如 `dir1`、`dir2`。

### 2. 单侧目录模式

```text
profile
|____attention
|    |____rank_xxx
|         |____ASCEND_PROFILER_OUTPUT
|              |____kernel_details.csv
|____ffn
     |____rank_xxx
          |____ASCEND_PROFILER_OUTPUT
               |____kernel_details.csv
```

在这种模式下，脚本会递归查找该侧目录下的所有 `kernel_details.csv`。

### 3. 单文件模式

直接输入某一个 `kernel_details.csv`。

## 统计规则

### Attention 侧

- 只统计每个 `E2a -> 下一次 A2e` 窗口内
- 所有非 marker kernel 的 `Duration(us)` 之和
- `A2e` / `E2a` 本身不计入阶段时间

### FFN 侧

- 只统计每个 `A2e -> 下一次 E2a` 窗口内
- 所有非 marker kernel 的 `Duration(us)` 之和
- `A2e` / `E2a` 本身不计入阶段时间

### 均值去极值规则

对所有 `mean_us` 相关统计，默认使用以下规则：

1. 先按 `Q3 + 3 * IQR` 剔除很大的上尾离群值
2. 再去掉 1 个最小值和 1 个最大值
3. 用剩余样本计算均值

这条规则同时用于：

- 阶段时间的 `mean_us`
- 额外算子统计列的 `<op>_mean_us`

## 输入参数

### 必选参数

```bash
python3 extract_kernel_stage_summary.py <input_path>
```

`input_path` 支持三种形式：

- `benchmark_result` 根目录
- `profile/attention` 或 `profile/ffn` 这样的单侧目录
- 某一个 `kernel_details.csv`

### 可选参数

```bash
-o, --output
```

- 自定义输出 CSV 路径
- 只在单文件模式或单侧目录模式下使用
- `benchmark_result` 模式下会固定输出两张表到当前目录

```bash
--format markdown|json
```

- 控制终端输出格式
- 默认 `markdown`

```bash
--attn-ops
```

- 指定 Attention 侧需要额外统计平均时延的算子列表
- 多个算子用逗号分隔
- 例如：`FusedInferAttentionScore,A2e,E2a`

```bash
--ffn-ops
```

- 指定 FFN 侧需要额外统计平均时延的算子列表
- 多个算子用逗号分隔
- 例如：`GroupedMatmul,A2e,E2a`

```bash
--workers
```

- 只在 `benchmark_result` 根目录模式下生效
- 当有多个实验目录时，使用多进程并行处理每个实验的统计
- 未指定时会自动选择合适的进程数

## 输出模式

### A. `benchmark_result` 根目录输入

例如：

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /path/to/benchmark_result \
  --attn-ops FusedInferAttentionScore,A2e,E2a \
  --ffn-ops GroupedMatmul,A2e,E2a
```

会在**当前目录**生成两张总表：

- `<benchmark_result_name>_attention_kernel_stage_summary.csv`
- `<benchmark_result_name>_ffn_kernel_stage_summary.csv`

如果有多个实验目录，脚本会默认使用多进程并行处理这些实验；也可以通过 `--workers` 手动指定并行进程数。

例如输入目录叫 `afd_result`，则输出：

- `afd_result_attention_kernel_stage_summary.csv`
- `afd_result_ffn_kernel_stage_summary.csv`

#### 表含义

- 每张表的每一行表示一个实验目录
- `attention` 表中每行表示一个实验目录下所有 Attention rank 合并后的统计结果
- `ffn` 表中每行表示一个实验目录下所有 FFN rank 合并后的统计结果

#### 输出列

基础列：

- `experiment`
- `side`
- `rank_count`
- `raw_count`
- `trimmed_count`
- `mean_us`
- `min_us`
- `max_us`
- `p75_us`
- `p90_us`
- `p99_us`
- `trim_rule`

如果传入算子列表，还会追加动态列：

- `<op>_mean_us`

例如：

- `FusedInferAttentionScore_mean_us`
- `A2e_mean_us`
- `E2a_mean_us`

### B. 单侧目录输入

例如：

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /path/to/profile/attention \
  --attn-ops FusedInferAttentionScore,A2e,E2a
```

默认会在该目录下生成：

- `kernel_stage_summary.csv`

#### 表含义

- `scope=rank`：每个 rank 1 行
- `scope=overall`：该侧所有 rank 合并后的汇总 1 行

#### 输出列

- `scope`
- `rank_name`
- `csv_path`
- `side`
- `stage_name`
- `rank_count`
- `raw_count`
- `trimmed_count`
- `mean_us`
- `min_us`
- `max_us`
- `p75_us`
- `p90_us`
- `p99_us`
- `trim_rule`

以及可选动态列：

- `<op>_mean_us`

### C. 单文件输入

例如：

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /path/to/kernel_details.csv
```

默认会在同目录下生成：

- `kernel_details_stage_summary.csv`

输出表只有 1 行，对应这个文件本身。

## 使用示例

### 示例 1：按实验根目录输出两张总表

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /Users/jiangchenzhou/Desktop/afd_result
```

### 示例 2：根目录模式，同时统计 Attention 算子均值

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /Users/jiangchenzhou/Desktop/afd_result \
  --attn-ops FusedInferAttentionScore,A2e,E2a
```

### 示例 3：根目录模式，同时统计 Attention 和 FFN 的指定算子

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /Users/jiangchenzhou/Desktop/afd_result \
  --attn-ops FusedInferAttentionScore,A2e,E2a \
  --ffn-ops GroupedMatmul,A2e,E2a
```

### 示例 3.1：根目录模式，显式指定并行进程数

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /Users/jiangchenzhou/Desktop/afd_result \
  --workers 4
```

### 示例 4：只分析某个 attention 目录

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /Users/jiangchenzhou/Desktop/afd_result/BSIZE_80_12A4F20260326_133604/profile/attention
```

### 示例 5：只分析某一个 `kernel_details.csv`

```bash
python3 /Users/jiangchenzhou/code/afd-skills/ascend-afd-profile/scripts/extract_kernel_stage_summary.py \
  /Users/jiangchenzhou/Desktop/afd_result/BSIZE_80_12A4F20260326_133604/profile/attention/.../kernel_details.csv \
  --format json
```

## 终端输出

脚本除了写 CSV 之外，也会在终端输出摘要：

- `benchmark_result` 模式：输出两张总表路径，以及每个实验的简要统计
- 单侧目录 / 单文件模式：输出当前输入对应的统计摘要

## 当前实现总结

这个脚本现在支持：

- `benchmark_result` 根目录聚合
- 单侧目录聚合
- 单文件分析
- 动态算子均值列
- 大极值过滤后的均值统计
- Attention / FFN 两侧分别输出
