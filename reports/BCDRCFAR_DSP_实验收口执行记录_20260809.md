# BCDRCFAR-DSP 实验收口执行记录

更新日期：2026-08-09

## 本轮目标

按 2026-08-08 的收口计划继续推进，优先处理 Birmingham 626 fresh-domain 缺口；若无法形成 detector validation，则把 626 收束为可复现 raw-data boundary，并冻结论文实验主张。

## 本轮新增证据

新增脚本：

- `experiments/summarize_birmingham_626_stare4_all_prefix_chunks.py`

新增产物：

- `results/dsp_v3_public_data/birmingham_626/stare_4_all_prefix_chunk_summary.json`
- `results/dsp_v3_public_data/birmingham_626/stare_4_all_prefix_chunk_summary.md`

核心结果：

- cached decompressed member prefix：`8,388,608` bytes
- 从 seed node `0x37113a` 追溯到左端 sibling，并向右遍历 cached-prefix 内可达 B-tree sibling chain
- cached-prefix 内可识别 `3` 个 `TREE` raw-data chunk nodes
- 三个节点共 `171` 条 chunk records
- 其中 `148` 条 chunk payload 在当前 prefix 内完整覆盖并成功 zlib 解压
- `23` 条记录因 payload 超出当前 cached prefix，标记为 `outside_cached_prefix`
- 解压错误数：`0`
- 每个成功 chunk 解压后均为 `59,392` bytes，即 `3,712` 个 complex double pairs
- decoded chunk 的 `offset1` 覆盖 `[0, 147]`

这比 2026-08-08 的单节点 `57` 条 chunk 证据更强：现在已证明 `stare_4_radar.mat` 的 raw-data chunk index 可沿 sibling chain 重建，且至少前 `148` 个时序 chunk 可直接恢复为数值 IQ 块。

## 判定

`Birmingham 626` 仍不能升级为 detector-validation endpoint。

理由：

- 当前只恢复了 `stare_4_rx1` 的前缀 chunk，而不是完整 acquisition。
- 仍没有完整 target-exclusion mask、scene-level independent acquisition protocol、以及和冻结 BC-DRCFAR endpoint 对齐的盲测输入表。
- 626 文件级结构证明已明显加强，但尚未达到 fresh-domain confirmatory experiment 的最低证据要求。

因此，论文实验主张应冻结为：

`BC-DRCFAR improves acquisition-level false-alarm reliability under retrospective external acquisition shifts, with Birmingham 626 providing a reproducible raw-data access boundary rather than an independent detector-validation result.`

## 论文实验收口口径

主线可写结论：

- 发展集与回顾性外部 IPIX acquisition-disjoint 结果支持 feature-head/background-conditioned calibration 的 false-alarm reliability 改善。
- 低秩与 grouped calibration 是机制/对照补充，不应替代 feature head 作为主叙事。
- 时间 holdout 结果显示 feature head 能缓解漂移，但不能消除长期时间退化。
- fold3/fold4 是稳定性边界，不应包装成模型失败由样本配比导致。
- Birmingham 626 是 raw-data reproducibility boundary：可证明远程 ZIP、MAT/HDF5 schema、B-tree chunk index、chunk payload 解压链路，但不能作为 fresh-domain confirmatory validation。

不应写的结论：

- 不写“626 完成独立验证”。
- 不写“BC-DRCFAR 已解决跨域 CFAR”。
- 不写“新数据集上 detector validation 成功”。
- 不把 626 chunk-level 可读性等同于 scene-level detection performance。

## 已验证命令

运行脚本：

```powershell
& 'D:\百度网盘拉取包\dsp升级\.venv-bcdrcfar-gpu\Scripts\python.exe' -u 'experiments\summarize_birmingham_626_stare4_all_prefix_chunks.py'
```

验证测试：

```powershell
py -m pytest tests\test_bcdrcfar_evaluation.py tests\test_confirmation_guard.py tests\test_mat_radar.py
```

结果：

- `9 passed in 18.80s`

注意：

- `.venv-bcdrcfar-gpu` 内未安装 `pytest`，测试使用系统 `py` 环境完成。
- 仓库目录不是 git repository，本轮以文件产物与命令输出作为可复现实验证据。

## 最终执行建议

现在应停止继续加模型变体，进入论文写作收口。

下一步只做三件事：

1. 把 `stare_4_all_prefix_chunk_summary` 纳入 Appendix / Evidence bundle。
2. 将正文 Results 聚焦到 IPIX feature-head、long-horizon holdout、ablation、failure boundary 四组结果。
3. 在 Discussion 明确写出 626 boundary：raw-data access reproducible, detector validation not claimed。
