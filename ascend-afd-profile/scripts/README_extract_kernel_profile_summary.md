# `extract_kernel_profile_summary.py`

用于把 normal kernel 和 AFD kernel 的统计合并到同一张 CSV。

脚本会递归扫描输入路径下的 `kernel_details.csv`，并按实验目录名过滤：

- 实验目录名包含 `normal`：按 normal kernel 逻辑统计，输出 `experiment_type=normal`、`role=normal`
- 实验目录名包含 `afd`：按 AFD kernel 逻辑统计，只保留 `attention` / `model_runner` / `ffn` 分支，输出 `experiment_type=afd`、`role=attn` 或 `role=ffn`

## 用法

```bash
python3 ascend-afd-profile/scripts/extract_kernel_profile_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --normal-ops 'MoeDistributeCombineV2,QuantBatchMatmulV3,SwiGlu' \
  --attn-ops 'FusedInferAttentionScore,A2e,E2a' \
  --ffn-ops 'GroupedMatmul,A2e,E2a' \
  -o ./kernel_profile_summary.csv
```

normal 侧仍支持 loop 模式：

```bash
python3 ascend-afd-profile/scripts/extract_kernel_profile_summary.py \
  /path/to/benchmark_results/deepseek-v3.2 \
  --normal-mode loop \
  --normal-loop-name mla_loop_total \
  --normal-ops-file ascend-afd-profile/scripts/mla_loop_ops.txt \
  --attn-ops 'FusedInferAttentionScore' \
  --ffn-ops 'GroupedMatmul'
```

为了兼容旧 normal 脚本的参数，也可以继续使用 `--ops`、`--ops-file`、`--mode`、`--loop-name`，它们分别等价于 `--normal-ops`、`--normal-ops-file`、`--normal-mode`、`--normal-loop-name`。

## 输出 CSV 字段

- `experiment_type`：`normal` 或 `afd`
- `role`：`normal`、`attn` 或 `ffn`
- `match_mode`：`op` 或 `loop`
- `scope`：`profile` / `experiment` / `overall`
- `experiment`
- `profile_name`
- `rank_name`
- `op_name`
- `csv_count`
- `sample_count`
- `mean_us`
- `pXX_us`：由 `--percentiles` 决定，默认 `p25_us,p50_us,p75_us,p90_us,p99_us`
