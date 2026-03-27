---
name: ascend-afd-profile
description: 分析 Ascend / 昇腾 NPU 上 AFD（Attention FFN Disaggregation）的性能 profile。用于读取 benchmark_result 实验目录，遍历每组实验的 scripts/run_params.txt、profile/attention 和 profile/ffn 产物，定位 Attention、FFN、handoff、通信、负载不均衡等瓶颈；也支持基于 kernel_details.csv 统计剔除通信算子后的 A/F 两侧计算耗时，用于比较 Attention 与 FFN 的实际计算时间差距。
---

# 昇腾 AFD Profile 分析

这个 skill 用于分析 Ascend / 昇腾 NPU 上 AFD 场景的 profile 数据。

默认输入是一个 `benchmark_result` 根目录(目录名任意)，目录结构约定如下：

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

其中：

- `dir1`、`dir2` 等表示不同实验记录
- `log/` 暂时不作为重点输入，除非用户明确要求结合日志一起分析
- `profile/attention/` 表示 Attention 侧 profile 存档
- `profile/ffn/` 表示 FFN 侧 profile 存档
- `scripts/run_params.txt` 表示该组实验配置，是后续做实验对比的主索引

当前 profile 分析时，优先递归查找两侧目录中的 `op_statistic.csv`，它是默认主分析输入。
如果用户明确希望“去除通信算子后比较 A/F 两侧纯计算时间”，则进一步读取 `kernel_details.csv`：

- 在 `profile/attention` 下，只统计每个 `E2a -> 下一次 A2e` 窗口中非通信 marker 算子的执行时间之和
- 在 `profile/ffn` 下，只统计每个 `A2e -> 下一次 E2a` 窗口中非通信 marker 算子的执行时间之和
- 这些窗口可近似视为每个 microbatch 在 A / F 两侧的本地计算时间，用于比较 Attention 与 FFN 的计算耗时差距

## 目标

收到一个 `benchmark_result` 路径后，按如下方式工作：

1. 遍历所有实验子目录，如 `dir1`、`dir2`
2. 读取每组实验的 `scripts/run_params.txt`
3. 分别读取 `profile/attention` 与 `profile/ffn` 下的 profile 产物
4. 在需要时，从 `kernel_details.csv` 中额外抽取去除通信算子的 A / F 两侧 microbatch 计算时间
5. 建立“实验配置 -> profile 结果 -> 瓶颈判断”的映射
6. 输出单实验分析和跨实验对比结论

当前 `run_params.txt` 的默认解析规则如下：

- 重点读取 `Test Scenario:` 之后的配置内容
- 只把其中的 `key=value` 对视为本次实验配置
- `Test Scenario:` 之前或之外的内容，默认作为背景文本，不纳入主配置对比
- 例如 `BSIZE=24` 表示每个 Attention 的 `BSIZE=24`

如果后续 `run_params.txt` 格式继续细化，再在这个基础上扩展：

- 优先保持 `Test Scenario:` 作为配置段起点
- 优先支持标准 `key=value` 形式
- 无法结构化的内容保留原文摘要
- 明确标记“已解析字段”和“未结构化字段”

## 预期输入

在开始分析前，优先确认或推断这些信息：

- `benchmark_result` 根目录路径
- 每个实验目录是否都遵循 `profile/attention`、`profile/ffn`、`scripts/run_params.txt` 约定
- 工作负载信息：prefill/decode、seq len、batch size、hidden size
- 并行配置：TP、DP、EP，或其他 AFD 相关调度参数
- AFD 拓扑：Attention 和 FFN 是否分离到不同 rank / 进程 / 设备
- 软件栈：CANN、`torch_npu`、框架版本

如果用户只要求“先做方案设计”，则不要假设 profile 文件细节，先给出扫描策略、字段抽取策略和分析输出格式。

## 工作流

1. 扫描实验目录。
   在 `benchmark_result` 下找所有一级实验目录，忽略明显无关目录。

2. 建立实验清单。
   对每个实验目录记录：
   - 实验名
   - `scripts/run_params.txt` 是否存在
   - `profile/attention` 是否存在
   - `profile/ffn` 是否存在
   - 哪些必要输入缺失

3. 解析实验配置。
   优先读取 `scripts/run_params.txt` 中 `Test Scenario:` 之后的 `key=value` 对，提取能用于对比的参数，比如：
   - batch size
   - seq len
   - 并行度
   - micro batch
   - AFD 相关开关
   - 模型规模
   - 测试模式或数据集标识

   默认不要把 `Test Scenario:` 之外的文本和配置项混在一起分析。

4. 分析 Attention 与 FFN profile。
   优先递归查找 `profile/attention` 和 `profile/ffn` 下的 `op_statistic.csv`。

   对关键算子的执行时间分析，默认同时统计：
   - 均值
   - 最小值
   - 最大值

   Attention 侧重点关注：
   - `A2e`
   - `E2a`
   - Attention 相关算子时延

   FFN 侧重点关注：
   - `A2e`
   - `E2a`
   - `GroupMatmul`
   - `MoeDispatch`
   - `MoeCombine`

   按侧别分别判断：
   - Attention 侧关键算子是否主导时延
   - FFN 侧关键算子是否主导时延
   - 两侧 `A2e` / `E2a` 时延是否存在明显不对称
   - 关键算子的均值、最小值、最大值是否显示出明显抖动或长尾
   - 是否能从这些关键算子推断当前瓶颈落在哪一侧

5. 在需要时分析 `kernel_details.csv` 中的 A / F 两侧计算时间。
   当用户关注“剔除通信算子后的纯计算时间”时，优先递归查找 `profile/attention` 和 `profile/ffn` 下的 `kernel_details.csv`。

   统计规则如下：
   - 如果文件位于 `profile/attention` 下，只统计每个 `E2a -> 下一次 A2e` 之间所有非 marker kernel 的 `Duration(us)` 之和
   - 如果文件位于 `profile/ffn` 下，只统计每个 `A2e -> 下一次 E2a` 之间所有非 marker kernel 的 `Duration(us)` 之和
   - `A2e` / `E2a` 本身不计入窗口总时间
   - 默认做一次轻量去极值：去掉 1 个最小样本和 1 个最大样本
   - 输出这些窗口时间的均值、最大值、最小值、P75、P90、P99

   用这些统计优先回答：
   - 去掉通信算子后，Attention 和 FFN 哪一侧的纯计算更重
   - 两侧计算时间差距是稳定存在，还是只在高分位数放大
   - 长尾主要来自计算本身，还是更像通信 / handoff 抖动

6. 判断 AFD 级别瓶颈。
   综合 Attention 和 FFN 两侧，判断问题更像是：
   - Attention 侧瓶颈
   - FFN 侧瓶颈
   - Attention/FFN handoff 瓶颈
   - 通信瓶颈
   - 配置导致的负载不均衡

7. 做跨实验对比。
   用 `run_params.txt` 作为主线，把实验分组比较，优先回答：
   - 哪些参数变化带来了性能改善或退化
   - 哪种 profile 现象和哪类配置最相关
   - 是否存在某个参数组合导致 Attention 和 FFN 失衡

8. 输出结果。
   输出单实验摘要、实验间对比表述、最可能的根因和下一步建议。

## 参考材料的使用方式

按需加载以下参考文件：

- 需要把 profile 文件名映射为分析问题时，读 [artifacts.md](./references/artifacts.md)
- 需要组织最终诊断报告时，读 [analysis-template.md](./references/analysis-template.md)
- 需要自动扫描 `benchmark_result` 并汇总关键算子统计时，运行 [extract_afd_profile_summary.py](./scripts/extract_afd_profile_summary.py)
- 需要从 `kernel_details.csv` 统计去掉通信算子后的 A / F 两侧计算时间时，运行 [extract_kernel_stage_summary.py](./scripts/extract_kernel_stage_summary.py)

## 脚本入口

优先使用下面的脚本做初步汇总：

```bash
python3 /path/to/ascend-afd-profile/scripts/extract_afd_profile_summary.py /path/to/benchmark_result
```

默认输出 Markdown 摘要，也支持：

```bash
python3 /path/to/ascend-afd-profile/scripts/extract_afd_profile_summary.py /path/to/benchmark_result --format json
```

脚本默认行为：

- 遍历 `benchmark_result` 下所有一级实验目录
- 从 `run_params.txt` 中只提取 `Test Scenario:` 之后的 `key=value`
- 在 `profile/attention` 和 `profile/ffn` 下递归查找 `op_statistic.csv`
- 汇总关键算子的 mean / min / max
- 生成按实验分组的 Attention / FFN 摘要和瓶颈提示

当用户需要比较去除通信算子后的 A / F 两侧计算时间时，使用：

```bash
python3 /path/to/ascend-afd-profile/scripts/extract_kernel_stage_summary.py /path/to/profile/attention/.../kernel_details.csv
python3 /path/to/ascend-afd-profile/scripts/extract_kernel_stage_summary.py /path/to/profile/ffn/.../kernel_details.csv
```

脚本默认行为：

- 自动从路径判断当前文件属于 `attention` 侧还是 `ffn` 侧
- `attention` 侧只统计 `E2a -> 下一次 A2e` 之间的非 marker kernel 时间之和
- `ffn` 侧只统计 `A2e -> 下一次 E2a` 之间的非 marker kernel 时间之和
- 默认去掉 1 个最小值和 1 个最大值
- 输出 `mean / min / max / p75 / p90 / p99`

## AFD 场景下的诊断重点

### 1. Attention 侧

适合关注这些问题：

- `op_statistic.csv` 中 `A2e`、`E2a`、Attention 相关算子的时延分布
- `A2e`、`E2a`、Attention 相关算子的均值 / 最小值 / 最大值
- Attention 相关算子是否构成主要耗时
- Attention 侧的 `A2e` / `E2a` 是否异常偏长

这里的“Attention 相关算子”由实际 `op_statistic.csv` 中的算子名匹配决定，优先识别明显属于 Attention 路径的算子。

### 2. FFN 侧

适合关注这些问题：

- `op_statistic.csv` 中 `A2e`、`E2a`、`GroupMatmul`、`MoeDispatch`、`MoeCombine` 的时延分布
- `A2e`、`E2a`、`GroupMatmul`、`MoeDispatch`、`MoeCombine` 的均值 / 最小值 / 最大值
- `GroupMatmul` 是否主导 FFN 主体计算
- `MoeDispatch` / `MoeCombine` 是否主导 FFN 侧路由与聚合开销
- FFN 侧的 `A2e` / `E2a` 是否异常偏长

### 3. AFD handoff

在当前规则下，优先使用 `A2e` / `E2a` 时延差异做快速判断：

- 如果某一侧的 `A2e` 或 `E2a` 时延明显更长，优先判断另一侧是当前时延瓶颈
- 也就是说，较长的 `A2e` / `E2a` 往往意味着该侧正在等待对侧，真正更慢的是对侧
- 这个判断属于高优先级经验规则；如果后续有更多 profile 证据冲突，需要显式说明冲突并降低置信度
- 如果均值不高但最大值明显偏大，需要额外提示可能存在抖动、偶发阻塞或长尾问题

### 4. 去通信后的 A / F 纯计算时间

适合关注这些问题：

- `kernel_details.csv` 中按 microbatch 切分后的本地计算窗口时间
- `profile/attention` 下 `E2a -> 下一次 A2e` 的时间分布
- `profile/ffn` 下 `A2e -> 下一次 E2a` 的时间分布
- 两侧去除通信 marker 后，计算时间的均值和高分位数差距

优先用来回答：

- Attention 和 FFN 哪一侧的纯计算更慢
- 两侧差距是中位附近就存在，还是只在 P90 / P99 拉开
- 如果 `op_statistic.csv` 看起来像通信热点，但 `kernel_details.csv` 显示纯计算差距并不大，需要明确提示“更像 handoff / 通信问题，而不是单侧计算问题”


## 证据标准

- 不要只根据一个热点表下结论
- 但在当前实验体系下，`op_statistic.csv` 是第一优先级输入
- 至少结合两个信号源，例如热点表加 timeline，或算子统计加通信统计
- 如果用户明确要求比较 A / F 计算耗时差距，优先结合 `op_statistic.csv` 和 `kernel_details.csv`
- 区分“观测事实”和“根因推断”
- 每个结论都尽量给出高 / 中 / 低置信度
- 如果信息不足，明确指出最缺哪一类文件或哪一组实验对照
- 汇总关键算子时，优先同时报告均值、最小值、最大值，而不是只报单个平均值

## 默认输出结构

除非用户要求更长报告，否则优先使用下面的结构：

```markdown
## 实验扫描结果

- 共发现多少组实验
- 每组实验的配置文件和 profile 完整性

## 单实验结论

### dir1
- 配置摘要
- Attention 侧观察
- FFN 侧观察
- 去通信后的 A / F 计算时间对比
- 综合判断

### dir2
- ...

## 跨实验对比

- 哪些配置变化最关键
- 哪些指标或现象同步变化

## 最可能的根因

1. ...
2. ...
3. ...

## 下一步建议

1. ...
2. ...
3. ...
```

## 缺省策略

当 `run_params.txt` 内容还未细化时：

- 先把它当作“实验标签和配置来源”
- 优先抽取 `Test Scenario:` 之后可比较的稳定字段
- 对未知字段保留原始文本，不强行解释
- 如果两个实验 profile 很像但配置不同，明确指出“配置差异已观测，但尚未建立因果”

当 `profile/attention` 或 `profile/ffn` 中的文件组织暂时不统一时：

- 先递归查找 `op_statistic.csv`
- 如果一个侧别下存在多个 `op_statistic.csv`，需要说明它们分别来自哪个子目录
- 优先基于 `op_statistic.csv` 提取关键算子时延
- 关键算子时延默认汇总为均值 / 最小值 / 最大值
- 如果还有其他文件，再补充 timeline、`op_summary*.csv`、`kernel_details*.csv`、通信相关文件
- 缺失某类文件时，降低结论置信度，但仍给出现有证据下的最佳判断
