# Profile 产物参考

这个参考文件用于把 Ascend profile 目录中的文件，映射到它能回答的分析问题上。

## 当前主文件

- `op_statistic.csv`
  当前实验体系下的第一优先级输入。需要在 `profile/attention` 和 `profile/ffn` 下递归查找。主要用于提取关键算子的时延，并汇总均值、最小值、最大值，再做单实验与跨实验对比。

## 其他常见文件

- `op_summary*.csv`
  用于看热点算子、调用次数、总耗时、平均耗时。适合做热点排序，但不能单独用来判断关键路径。

- `kernel_details*.csv`
  用于看更细粒度的 kernel 级耗时、重复短 kernel、launch 规律，以及是否存在明显碎片化。

- timeline 或 trace 导出
  用于看 stream overlap、空泡、host-device 间隔、Attention 与 FFN 之间的边界，以及 handoff 是否串行化。

- `communication.json`
  用于分析分布式通信行为，特别是使用 `torch_npu.profiler` 并开启相应 profiler level 时产生的通信数据。

- `communication_matrix.json`
  用于看 rank 间通信模式和通信偏斜。

- `data_preprocess.csv`
  用于判断 CPU 侧或数据预处理是否拖慢了 NPU 执行。

- `profiler_info.json`
  用于确认 rank、worker、采集上下文等元信息。

## 在当前 benchmark_result 结构中的使用建议

### `profile/attention`

优先回答：

- `A2e` 的时延是多少
- `E2a` 的时延是多少
- Attention 相关算子的时延是多少
- 这些关键算子的均值、最小值、最大值分别是多少
- Attention 侧关键算子是否构成主要耗时

### `profile/ffn`

优先回答：

- `A2e` 的时延是多少
- `E2a` 的时延是多少
- `GroupMatmul` 的时延是多少
- `MoeDispatch` 的时延是多少
- `MoeCombine` 的时延是多少
- 这些关键算子的均值、最小值、最大值分别是多少
- FFN 侧关键算子是否构成主要耗时

## 单实验建议阅读顺序

对于某个实验目录，建议按下面顺序看：

1. `scripts/run_params.txt`
2. 递归查找 `profile/attention` 下的 `op_statistic.csv`
3. 递归查找 `profile/ffn` 下的 `op_statistic.csv`
4. 提取两侧关键算子时延
   并同步整理均值、最小值、最大值
5. 对比两侧 `A2e` / `E2a`
6. 需要时再补读 timeline、`op_summary*.csv`、`kernel_details*.csv`
7. 如果有通信文件，再补读 `communication.json` 和 `communication_matrix.json`

## 跨实验建议阅读顺序

当要比较多个实验时，建议按下面顺序推进：

1. 先对所有 `run_params.txt` 做字段归纳
2. 汇总各实验 Attention / FFN 两侧关键算子时延
3. 选出关键配置差异最大的实验组
4. 比较这些实验在 Attention 与 FFN 两侧的关键算子组成
5. 再比较 timeline 中的空泡、串行化、等待和通信重叠情况

## 每类文件主要回答什么问题

- Attention 慢，还是 FFN 慢？
  先比两侧 `op_statistic.csv` 中的关键算子时延，特别是 `A2e` / `E2a`，再决定是否需要补充 timeline。

- 是否存在明显抖动或长尾？
  先比关键算子的均值、最小值、最大值；如果最大值远大于均值，需要提示存在长尾风险。

- 是算不满，还是在等待？
  先看关键算子是否集中，再结合 timeline 或 `kernel_details*.csv` 判断。

- 是通信问题吗？
  先看通信文件，再确认它是否真的落在关键路径，并检查是否缺乏 overlap。

- 是配置导致的不均衡吗？
  先比 `run_params.txt`，再比 `op_statistic.csv` 中关键算子时延是否随配置变化同步改变。

- 哪一侧是真正的时延瓶颈？
  如果某一侧的 `A2e` 或 `E2a` 明显更长，优先判断另一侧是当前时延瓶颈。

## 常见误区

- 只看总耗时最高的算子，不看它是否真的在关键路径上
- 看到某一侧的 `A2e` / `E2a` 更长，就误以为这一侧更慢；在当前经验规则下，往往意味着另一侧才是瓶颈
- 把所有通信都当成纯开销，不检查它是否已被计算掩盖
- 看到 kernel 效率低就直接归因给算子实现，而忽略 shape 太小
- 只看 Attention 或只看 FFN，不把两侧放在同一实验配置下联合判断
- 只看某一组实验，不做配置对照就下结论
