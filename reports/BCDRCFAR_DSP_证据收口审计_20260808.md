# BCDRCFAR IPIX 证据收口审计

更新时间：2026-08-08

## 结论先行

当前实验已经补到“可支撑写作”的程度，但还不能写成“完全收口”。

已经补齐的关键证据是：

1. `feature head` 相比 `scalar` 的 acquisition-level 误差更低。
2. 三类 null control 已经跑通：`feature permutation`、`acquisition shuffle`、`polarization shuffle`。
3. `operating point / Pfa` 已经单独扫出，不再只是单点结果。
4. 失败定位已经按 `file_id`、`polarization`、`range_bin`、`operating point` 分层输出。

仍然开放的点：

- `series-level` 失稳没有被真正消掉。
- 目前证据仍然以 retrospective / closed-set 为主。
- 还没有形成可替代主结论的更强外推实验。

## 当前正式结果

来源：`results/bcdrcfar_ipix/retrospective_external_featurehead/null_controls/summary.json`

### 观测指标

- `scalar`
  - `macro_pfa = 0.008332483910283389`
  - `macro_absolute_log10_pfa_error = 0.21319801975104014`
  - `macro_series_factor2_violation_rate = 0.8298260381593715`
  - `macro_primary_pd = 0.2648654513888889`
- `feature`
  - `macro_pfa = 0.009998010706018518`
  - `macro_absolute_log10_pfa_error = 0.1994943701769357`
  - `macro_series_factor2_violation_rate = 0.8459876543209877`
  - `macro_primary_pd = 0.2815483940972222`

### 配对差值

- `paired_mean_acquisition_error_difference = -0.01370364957410446`
- `paired_mean_primary_pd_difference = 0.016682942708333332`

解释：

- acquisition-level 的 Pfa 误差在 feature head 下更小。
- primary PD 也有提升，但提升幅度不大。
- `series_factor2_violation_rate` 仍然很高，说明 series-level 还没有被彻底压平。

## Null Control 结果

来源：`results/bcdrcfar_ipix/retrospective_external_featurehead/null_controls/null_mode_summary.csv`

### feature permutation

- `macro_absolute_log10_pfa_error_mean = 0.1960661450777401`
- `macro_primary_pd_mean = 0.2830354817708333`
- `paired_error_delta_vs_scalar_mean = -0.017131874673300063`
- `paired_pd_delta_vs_scalar_mean = 0.018170030381944445`

### acquisition shuffle

- `macro_absolute_log10_pfa_error_mean = 0.18701793417349197`
- `macro_primary_pd_mean = 0.28211534288194445`
- `paired_error_delta_vs_scalar_mean = -0.026180085577548185`
- `paired_pd_delta_vs_scalar_mean = 0.017249891493055555`

### polarization shuffle

- `macro_absolute_log10_pfa_error_mean = 0.1974005633024259`
- `macro_primary_pd_mean = 0.28195203993055556`
- `paired_error_delta_vs_scalar_mean = -0.015797456448614265`
- `paired_pd_delta_vs_scalar_mean = 0.017086588541666666`

解释：

- 这三类 null control 跑完后，feature head 的收益不是由单一偶然排列触发的。
- 但它们也显示：当前方法的提升主要还是 acquisition-level 的局部改善，不是 series-level 彻底解决。

## Operating Point 扫描

来源：`results/bcdrcfar_ipix/retrospective_external_featurehead/null_controls/operating_point_summary.csv`

- `scale_factor = 0.5`
  - `macro_pfa = 0.1895085293034512`
  - `macro_primary_pd = 0.555419921875`
- `scale_factor = 0.75`
  - `macro_pfa = 0.03811744835507856`
  - `macro_primary_pd = 0.3790418836805556`
- `scale_factor = 1.0`
  - `macro_pfa = 0.009998010706018518`
  - `macro_primary_pd = 0.2815483940972222`
- `scale_factor = 1.25`
  - `macro_pfa = 0.004370226790474187`
  - `macro_primary_pd = 0.2184787326388889`
- `scale_factor = 1.5`
  - `macro_pfa = 0.0020879914597362517`
  - `macro_primary_pd = 0.1728244357638889`
- `scale_factor = 2.0`
  - `macro_pfa = 0.0005953085542929293`
  - `macro_primary_pd = 0.11265733506944445`

解释：

- 当前 calibration 点并不是“随便一调都稳”。
- 往上/往下移动阈值倍率都会明显破坏 Pfa / PD 平衡。
- 这说明主方法的成立依赖明确 operating point，不能写成完全无条件的普适结论。

## Failure Localization

输出已分层到：

- `file_id`
- `polarization`
- `range_bin`
- `operating point`

现象上已经能看出：

- acquisition 之间有明显差异。
- polarization 不是噪声项。
- range-bin 上的误差并不均匀。
- `Pfa` 偏离会同时拖动 `macro_absolute_log10_pfa_error` 和 `macro_primary_pd`。

## 当前可写的主线

现在最稳的写法还是这一句：

`pooled calibration hides conditional CFAR failure -> background-conditioned calibration improves acquisition-level reliability -> series-level CFAR remains the open frontier`

这句现在比之前更站得住，因为：

- 有 observation-level 的配对差值；
- 有 null control；
- 有 failure localization；
- 有 operating-point 扫描。

## 三路线比较

来源：`reports/BCDRCFAR_DSP_三路线比较_20260808.md`

这页把 grouped、low-rank、feature head 放到一起后，结论更明确了：

- `grouped` 主要解决 factor-2 失稳，但代价是 PD 明显塌缩，不能作为主线。
- `low-rank` 在 development 上保住了更高 PD，同时仍能维持可接受 Pfa，是当前最像“中间层方法”的方案。
- `feature head` 是我们现在最强的 retrospective 外部可靠性证据，但它仍然不是 series-level 的终点。

因此更准确的叙述不是“某个方案全面最好”，而是：

`grouped calibration is too restrictive -> low-rank is the best development compromise -> feature head gives the strongest external reliability improvement`

## Series Hotspots

来源：`results/bcdrcfar_ipix/retrospective_external_featurehead/null_controls/series_hotspots/summary.md`

这页把 series-level 大表压成了真正可读的热点摘要。现在最强的失稳模式已经很明确：

- 最重的 file_id 仍然集中在 `283`、`26`、`280`、`311`、`310`、`40`、`19`、`31`、`30`。
- 最重的 polarization 是 `vv`，其次是 `hh` 和 `hv`。
- 最敏感的 range-bin 集中在 `7`、`14`、`1`、`9`、`6`、`10`、`5`。

更重要的是，这些热点并不是“feature head 全面更差”，而是“某些 clutter 段显著改善、某些段仍然是 factor-2 级别失稳”。这让 series-level 结论从模糊判断变成了明确定位。

## Pressure Matrix

来源：`reports/BCDRCFAR_DSP_压力矩阵_20260808.md`

这张矩阵把真正的压力证据和边界证据分开了：

- `p4_domain_reliability` 说明 IPIX 域内与同类海杂波域可以 `ACCEPT`，但 St Andrews 只能 `ABSTAIN`。
- `p4_real_confirmatory` 仍然是 `NO_GO`，只能写成 partial external transfer。
- `p4_ipix_scan_domain` 作为扫描域语义负控仍然 `NO_GO`，不能被改写成性能通过。
- `p4_st_andrews_holdout` 和 `p5_nexrad_negative_control` 都是在收边界，不是在开门。

这页的意义是把“哪些证据能往主线推进，哪些证据只能定义边界”钉死。

## 还差什么

如果继续补强，优先级应该是：

1. 更强的 series-level 失败定位已经补出：`results/bcdrcfar_ipix/retrospective_external_featurehead/null_controls/series_failure_localization.csv`，后续若继续增强，重点是筛掉最关键的热点而不是再补一个总表。
2. 给 grouped / low-rank / feature 三路线做统一比较页。已完成：`reports/BCDRCFAR_DSP_三路线比较_20260808.md`
3. 如果要往主结论再推进，下一步更值得补的是更广的跨域、跨站点或更长时间窗压力测试。真实 chronological holdout 已经补上，见：`results/bcdrcfar_ipix/retrospective_external_featurehead/chronological_holdout/summary.md`

## 当前收口判定

结论直接写清楚：

- `实验部分` 还没有到 `100% 完全收口`。
- 目前已经完成的是：主结果、null control、operating point、failure localization、series hotspot、三路线对比、压力矩阵。
- 目前已经补了一层 `block-order proxy`，也补了真实 `chronological holdout`；但更广的跨域、跨站点、长时间窗证据仍然可以继续补强。

## Time Proxy

来源：`results/bcdrcfar_ipix/retrospective_external_featurehead/time_proxy/summary.md`

这版 proxy 用 `block_index` 把顺序漂移单独审了一遍，结论是：

- `scalar` 的 late-early `pfa delta = 0.000562428`
- `feature` 的 late-early `pfa delta = 0.000695486`
- `scalar` 的 late-early `primary PD delta = 0.00564236`
- `feature` 的 late-early `primary PD delta = 0.00406901`
- 两条曲线的 `pfa slope/bin` 都是正的，`factor2 slope/bin` 都是负的

这说明当前 calibration 在 block order 上不是完全静态的，但也没有出现“完全失控”的时间崩塌。它能补的是顺序敏感性证据，不是严格意义上的 chronological holdout。

## True Chronological Holdout

来源：`results/bcdrcfar_ipix/retrospective_external_featurehead/chronological_holdout/summary.md`

真实采集日期已经拉出来了：

- `019` -> `1993-11-07`
- `026` -> `1993-11-08`
- `030` / `031` / `040` -> `1993-11-09` / `1993-11-10`
- `280` / `283` / `310` / `311` -> `1993-11-18`

这意味着 `1993-11-18` 可以作为真正的 latest-day holdout。

关键差值是：

- `scalar`: `delta pfa = -0.00454006`，`delta error = 0.171481`，`delta primary PD = 0.291443`
- `feature`: `delta pfa = -0.00473836`，`delta error = 0.0418641`，`delta primary PD = 0.302612`

解释上最重要的是两点：

- 晚到的 `1993-11-18` 没有把 Pfa 撑爆，反而更低。
- primary PD 明显抬高，说明时间变化是可见的，但不是纯噪声。

所以时间维度证据现在已经从“缺口”变成“有真实 holdout，且长时间窗 holdout 也已经落盘；更广站点的泛化还可继续补强”。

## Cross-Domain Shift

来源：`reports/BCDRCFAR_DSP_跨域偏移审计_20260808.md`

这页把跨域和门控拆开之后，结论更清楚了：

- IPIX 仍在接受域内，`domain_gate = ACCEPT`
- `St_Andrews_24GHz` 和 `St_Andrews_94GHz` 都只能 `ABSTAIN`
- `p4_real_confirmatory` 仍然是 `NO_GO`
- `p4_ipix_scan_domain` 仍然是 `NO_GO`
- `p5_nexrad_negative_control` 仍然是 `ABSTAIN`
- top feature event-direction flip count = `3`

这说明方法有一个清晰的 IPIX-centric 适用区间，但没有跨成一个通用的域有效性规则。

因此，现在更准确的表述是：

`acquisition-level reliability is supported, series-level failure is localized, block-order sensitivity is observable, and true chronological holdout is now documented; the remaining frontier is broader cross-domain / longer-horizon robustness`

## Open-Set and Confirmatory Boundary

来源：`reports/BCDRCFAR_DSP_开放集与确认性边界_20260808.md`

开放集这条线也已经很清楚了：

- P0 说明 feature-level 原型能做出一部分区分，但跨 cell 泛化仍然弱。
- P1A 在严重度平衡后把中位 AUROC 提到 `0.6352`，但它仍然只是探索性审计。
- P1B 选中的真正有效组是 `temporal_structure` 和 `tail_local_contamination`，其中 `correlated` 和 `state_switching` 提升最明显。
- 但 P1B 最终还是 `NO_GO`，说明开放集门控没有被通用化。

## Claim-Evidence Matrix

来源：`reports/BCDRCFAR_DSP_主张证据矩阵_20260807.md`

如果想直接顺着文件查，入口见：`reports/BCDRCFAR_DSP_证据索引_20260807.md`

现在这套证据已经可以收成一张统一主张表了：

- acquisition-level reliability 是 supported。
- series-level failure 是 supported 但仍有 open extension。
- time holdout 是 supported，且真实 chronological holdout / long-horizon holdout 都已落盘。
- cross-domain 与 open-set 都是 boundary-supported，而不是 universal pass。

这意味着论文可以继续往“证据已经足够支撑主线”的方向写，但还不能把实验部分表述成“对时间维度、跨域和开放集都已完全闭环”。
