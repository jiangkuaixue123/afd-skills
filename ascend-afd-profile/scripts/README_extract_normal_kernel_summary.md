# extract_normal_kernel_summary.py

用于统计 normal 场景下 `kernel_details.csv` 中目标算子的耗时分布。

脚本支持两种模式：

- `op`：按单个算子匹配，统计这些算子的 `Duration(us)` 分布
- `loop`：按给定的有序算子序列匹配完整循环，统计每个循环总耗时的分布

脚本会递归扫描输入路径下所有名为 `kernel_details.csv` 的文件，支持多个实验目录、每个实验目录下多个 profile，并默认启用多进程并行。

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

也支持更扁平的布局，例如：

```text
experiment/
  profile/
    model_runner/
      rank0/
        ASCEND_PROFILER_OUTPUT/
          kernel_details.csv
```

脚本会自动从路径中识别：

- `experiment`
- `profile_name`
- `rank_name`

如果目录里没有显式 profile 名，会写成 `DEFAULT_PROFILE`。

## 匹配规则

可以通过 `--ops` 或 `--ops-file` 传入目标列表。

- `--ops`：命令行里直接传，多个项目用逗号分隔
- `--ops-file`：从文件读取，适合很长的循环序列

匹配时会先做名字归一化：

- 转小写
- 去掉非字母数字字符

然后按以下规则匹配：

- 完全匹配
- 或者 kernel 名以目标名为前缀

例如：

- `QuantBatchMatmulV3` 可以匹配 `QuantBatchMatmulV3_ND_NZ_int8_int8_bf16_high_performance_24`
- `SwiGlu` 可以匹配 `SwiGlu_3_high_performance_27`

在 `loop` 模式下，`--ops` 的顺序有意义，脚本会按顺序寻找一个完整循环；只有完整匹配到整套序列，才会记为一个样本。

如果使用 `--ops-file`，文件中的顺序同样会被保留，因此在 `loop` 模式下建议优先使用文件。

## `--ops-file` 文件格式

推荐格式：每行一个算子或前缀。

例如：

```text
# mla loop
AddRmsNorm_3_high_performance_33
Add_ce3d1ff8ba23a0bcb292da3577c1625d_high_performance_221000002
mla_preprocess_0_mix_aic
MatMulV2_ND_ND_FP16_FP16_false_true_all_98513
```

支持规则：

- 空行会被忽略
- 以 `#` 开头的行会被忽略
- 也支持一行里写多个，用逗号分隔

例如下面这种也可以：

```text
OpA,OpB,OpC
OpD
```

注意：

- `--ops` 和 `--ops-file` 只能二选一，不能同时传
- 在 `loop` 模式下，文件顺序就是循环匹配顺序

## 分位数

通过 `--percentiles` 指定要输出的分位数，例如：

```bash
--percentiles '50,80,90,95,99'
```

默认分位数是：

```text
25,50,75,90,99
```

## 基本用法

### 1. `op` 模式，统计一个 benchmark 根目录下所有实验

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

### 2. `loop` 模式，统计每个完整循环总耗时

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --mode loop \
  --loop-name mla_loop_total \
  --ops 'AddRmsNorm_3_high_performance_33,Add_ce3d1ff8ba23a0bcb292da3577c1625d_high_performance_221000002,mla_preprocess_0_mix_aic,MatMulV2_ND_ND_FP16_FP16_false_true_all_98513'
```

### 3. 使用 `--ops-file` 读取长序列

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --mode loop \
  --loop-name mla_loop_total \
  --ops-file /path/to/mla_loop_ops.txt
```

### 4. 只统计一个实验目录

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2/normal_bs24_dp32_tp1_in4096_20260409_143641 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

### 5. 只统计一个具体的 `kernel_details.csv`

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/kernel_details.csv \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu'
```

## 多实验、多 profile、并行处理

当输入路径下有很多实验目录、每个实验下又有多个 profile 时，脚本会递归扫描全部 `kernel_details.csv`，并按以下层级汇总：

- `profile`
- `experiment`
- `overall`

你可以通过 `--workers` 指定并行进程数：

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --mode loop \
  --loop-name mla_loop_total \
  --ops 'AddRmsNorm_3_high_performance_33,Add_ce3d1ff8ba23a0bcb292da3577c1625d_high_performance_221000002,mla_preprocess_0_mix_aic,MatMulV2_ND_ND_FP16_FP16_false_true_all_98513' \
  --scopes experiment,overall \
  --workers 8
```

说明：

- 默认会自动选择并发数
- `--workers 1` 表示关闭并行
- 脚本优先使用多进程
- 如果当前环境不允许创建进程池，会自动回退到线程池
- 如果某个目录下的 `kernel_details.csv` 解析失败，脚本会继续处理其他目录，并打印失败目录和原因

## 只输出某些层级

通过 `--scopes` 控制输出哪些层级，多个值用逗号分隔。

可选值：

- `profile`
- `experiment`
- `overall`

示例：

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --mode loop \
  --loop-name mla_loop_total \
  --ops 'AddRmsNorm_3_high_performance_33,Add_ce3d1ff8ba23a0bcb292da3577c1625d_high_performance_221000002,mla_preprocess_0_mix_aic,MatMulV2_ND_ND_FP16_FP16_false_true_all_98513' \
  --scopes overall
```

## 指定输出文件

```bash
python3 ascend-afd-profile/scripts/extract_normal_kernel_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --output ./normal_kernel_summary.csv
```

## 输出 CSV 字段说明

输出 CSV 包含以下列：

- `match_mode`：`op` 或 `loop`
- `scope`：统计层级，取值为 `profile` / `experiment` / `overall`
- `experiment`：实验目录名
- `profile_name`：profile 名
- `rank_name`：rank 目录名
- `op_name`：`op` 模式下是目标算子名；`loop` 模式下是 `--loop-name`
- `csv_count`：该层级汇总了多少个 `kernel_details.csv`
- `sample_count`：命中的样本数
- `mean_us`
- `pXX_us`：由 `--percentiles` 决定

在 `op` 模式下，`sample_count` 表示命中的 kernel 数。

在 `loop` 模式下，`sample_count` 表示识别出的完整循环数。

## CSV 排序规则

输出 CSV 按以下顺序排序：

1. `match_mode`
2. `op_name`
3. `scope`，顺序为 `overall -> experiment -> profile`
4. `experiment`
5. `profile_name`
6. `rank_name`

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
  --mode loop \
  --loop-name mla_loop_total \
  --ops 'AddRmsNorm_3_high_performance_33,Add_ce3d1ff8ba23a0bcb292da3577c1625d_high_performance_221000002,mla_preprocess_0_mix_aic,MatMulV2_ND_ND_FP16_FP16_false_true_all_98513' \
  --percentiles '50,90,99' \
  --scopes experiment,overall \
  --workers 8
```

## 注意事项

- 输入路径既可以是目录，也可以是单个 `kernel_details.csv`
- 如果给定算子没有匹配到任何记录，脚本会报错退出
- `loop` 模式会忽略不完整循环
- 当前分位数统计直接基于全部命中样本，不做去极值处理
- 如果部分目录解析失败，终端会打印失败目录及错误原因；JSON 输出里会带 `skipped_csvs`
