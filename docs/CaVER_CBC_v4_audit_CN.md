# CaVER/CBC v4：本轮结果审计、指标定义与下一轮实验方案

## 结论先行

本轮证据支持一个较窄但有研究价值的判断：在当前声明的 synthetic reference reactor 中，两个 intent-matched、轨迹相近的 ego 候选确实可以产生不同的非 ego 风险；因此“同意图候选的级联风险差异”不是由论文人为臆造的空问题。当前证据不支持更强的算法结论：`full` 没有优于 `no_events`，`anchor_graph` 没有优于同预算的 `anchor_full`，WOMD 初始化集合没有形成可评价的 matched+legal+ego-safe+reference-resolved 分母，真实端到端时延也没有测量。

最稳妥的投稿定位是：**在有限响应预测预算和不完整、延迟的响应证据下，如何识别同意图候选之间具有决策意义的下游风险差异，并在事件边界证据不足时触发候选条件的重新预测。** 贡献点应是 decision-focused boundary discrimination and selective recomputation，而不是“发现交通级联”。

本轮新增代码只补齐了诊断、分母、场景簇不确定性、离线记录计时和保守的 latency-aware reuse gate；它没有把当前 synthetic 结果升级成真实安全证明，也没有声称 graph/event 组件已经获得增量增益。

## 1. 已有结果的科学解释

### 1.1 Synthetic stress set

来自 [synthetic gate](sandbox:/workspace/scratch/2ddeb61865ff/work/results_v3/cbc_phase0_v3/synthetic_cbc_gate_v3.json) 和 [GPU ablation](sandbox:/workspace/scratch/2ddeb61865ff/work/results_v3/gpu_ablation/cbc_study.json)：

| 方法 | resolved critical sign error | underresolution | near-boundary MAE | union-risk MAE | ego-safe safety regret |
|---|---:|---:|---:|---:|---:|
| `anchor_copy` | 100.00% | 100.00% | 0.3147 | 0.0996 | 0.1009 |
| `source` | 81.25% | 81.25% | 0.3109 | 0.0982 | 0.1009 |
| `anchor_full` | 21.875% | 0.00% | 0.2000 | 0.1012 | 0 |
| `anchor_graph` | 31.25% | 28.125% | 0.1309 | 0.0385 | 0 |
| `no_events` | 6.25% | 6.25% | 0.1551 | 0.0832 | 0 |
| `full` | 6.25% | 6.25% | 0.1778 | 0.0825 | 0 |

样本包含 120 个 held-out scene、120 个当前运动学匹配 scene、100 个 matched 且两条候选都 ego-safe 的 scene、32 个 reference-resolved critical pair，且每个 action 有 192 个 reference rollouts。

可支持的最强结论是：

1. `anchor_copy` 比 source 明显差，说明简单复制 anchor 会把候选条件的响应差异抹掉，并产生 false-safe 倾向。
2. `full` 和 `no_events` 在关键排序错误率上完全相同；事件特征的增量排序能力没有被证明。
3. `anchor_graph` 比 `anchor_copy` 好，但比 `anchor_full` 的关键排序错误率差；图的平均 union-risk MAE 较低，不能据此宣称图提升了关键排序。
4. 当前 `matched_ego_safe_safety_ranking_regret=0` 不能单独解释为规划器已经完美，因为它只在有效的 matched+ego-safe 子集上比较安全风险，而且当前 regret 没有加入进度、舒适性、路线合法性、fallback 和计算成本。

`anchor_graph_minus_anchor_copy` 的场景 bootstrap CI 为 `[-0.8486,-0.5172]`，只说明 graph 不是“原样复制”的弱基线；`anchor_graph_minus_anchor_full` 的独立增量比较没有被原始包完整测量，必须补做 equal-budget non-graph control。

### 1.2 WOMD initialized set

来自 [WOMD gate](sandbox:/workspace/scratch/2ddeb61865ff/work/results_v3/cbc_phase0_v3/womd_cbc_gate_v3.json) 和 [candidate design](sandbox:/workspace/scratch/2ddeb61865ff/work/results_v3/cbc_phase0_v3/test_candidate_design_v3.json)：

- 120 个 scene，但 fixed policy 只有 15 个 matched scene；105 个因 `EGO_DYNAMICS` 失败。
- matched 且两条候选都 ego-safe 的 scene 为 0。
- reference-resolved critical pair 为 0。
- 只有所有 pair 比较中的少量 flip，matched 子集没有可用于风险排序的 flip。
- adaptive observation-only policy 的 matched proxy yield 约 96.7%，但 `lawful_route_verified=false`，且名义几何 clear 不是 reference ego safety。

因此 `NOT_TESTABLE_WITH_CURRENT_EVIDENCE` 是正确结论。WOMD 当前可以审计观测历史、候选匹配和数据质量，不能由 logged future 识别未执行 ego 动作的真实反事实风险。

## 2. 当前代码对指标的真实实现

### 已有但定义需要拆分的指标

`caver/validation/cbc_study.py` 当前使用：

```text
true_delta = p_ref(early) - p_ref(late)
pred_delta = p_pred(early) - p_pred(late)
critical = abs(true_delta) >= 0.20
sign_error = (pred_delta * true_delta <= 0) OR abs(pred_delta) < 0.05
```

这把“方向错”和“没有分辨出来”放进同一个数。v4 的 `extended-audit` 把它们拆成：

- `sign_error`：reference 已分辨、预测也越过 detection deadband、但方向相反；
- `underresolution`：reference 已分辨、预测仍在 deadband 内；
- `critical_miss = sign_error OR underresolution`；
- 同时保留 all-critical 与 reference-resolved 两个分母。

### 部分实现但不能按论文措辞使用的指标

- `critical_pair_underresolution_rate`：原来只是 `abs(pred_delta)<0.05`，没有把 reference MC resolution 和 reuse decision 纳入分母。
- `matched_ego_safe_safety_ranking_regret`：只比较 predicted downstream risk 的 argmin 与 reference risk 的差，未包含完整 task objective，也没有 route legality 证据。
- `any_brake_guard_flip_rate` 与 `first_mode_disagreement_mean_s`：只是 paired rollout 的结构差异摘要，不能当作事件 detector recall、precision 或 lead time。
- `reference_contact_flip_rate`：有 paired latent-world 点估计，但原始报告没有逐 pair MC CI 和 scene-cluster CI。
- `source_rollout_reduction_fraction=0.6667`：是调用次数 proxy，不是 p95/p99 wall-clock speedup。

### 当前完全没有实现的证据

原始 record 没有：

- planning callback 起点到 release 的 total wall-clock p50/p95/p99；
- callback/100 plans、真实 rollout/100 plans、fallback 和 deadline miss；
- explicit reuse decision，因此不能从 `anchor_copy` 自动推导 false-safe anchor reuse；
- explicit event labels、predicted trigger time、事件漏检和提前量；
- route-legal candidate label；
- rare-risk 的按 TTC、slack、guard distance、policy family 分层 calibration。

这些缺失项现在会被 v4 审计明确标记为 `NOT_MEASURED`，不会用代理量补成伪结果。

## 3. 论文中应固定的指标定义

对候选对 $(a,b)$ 和非 ego pair $p$，令
$\Delta^{ref}=C_p^{ref}(a)-C_p^{ref}(b)$，预测差为 $\widehat\Delta$。

1. **Critical pair**：$|\Delta^{ref}|\ge\delta$。
2. **Reference resolved**：$|\Delta^{ref}|>r_{MC}$，其中 $r_{MC}$ 是基于 paired reference worlds 的保守 finite-MC 半径。
3. **Critical sign error**：预测越过 detection deadband 且 `sign(Δhat) != sign(Δref)`。
4. **Critical underresolution**：reference resolved，但 `|Δhat| < τ_det`。
5. **Critical miss**：sign error 或 underresolution。三者必须分开报告。
6. **False-safe anchor reuse**：系统 gate 接受 reuse，而 held-out reference 显示 critical contrast/flip；分母同时给 reused-pair 和 planning-decision 两种版本。
7. **Event recall / lead time**：必须有 held-out event onset 和预测 trigger。定义 `lead = t_reference - t_trigger`，正值表示提前触发，负值表示晚触发。只有 guard flip 没有 trigger 时间时，指标为 `NOT_MEASURED`。
8. **Ego-safe downstream regret**：仅在 intent-matched、dynamically feasible、route-legal 且两条候选都 reference ego-safe 的集合上，比较完整 task-risk objective；报告有效样本数、release、abstain 和 uncertified 分层。
9. **Reference pair flip**：同一初始状态和同一 latent draw 的 paired outcome flip；每个 pair 报 binomial MC interval，聚合报 scene-cluster bootstrap CI。它是 coupling-specific simulator diagnostic，不是人类个体因果概率。
10. **Yield**：`candidate → intent-matched → dynamic-feasible → legal → both-ego-safe → reference-resolved-critical`，任何未知 legality 都不能计入 legal。
11. **Calibration**：有限 MC 条件均值只能称 `reference-mean reliability/ECE`；binary Brier/ECE 需要逐 rollout Bernoulli label。按 risk bin、TTC/headway/slack、guard distance、event type 和 reference family 分层。

## 4. 为什么当前时延图没有体现价值

问题主要在机制和使用方式同时存在，而不是简单的“网络容量不够”。

1. source rollout 本身已经用 delayed history index 让 B/C 反应；graph 又注入 `delay_mean` 和离散 lag，存在 double counting 或 stale hidden state 风险。
2. `source_temporal_graph` 用所有 source rollout 的均值，`delay_mean` 是单个 agent 的均值；没有 delay quantile、event-time distribution 或 candidate/anchor 的 time shift。
3. `DelayedGraphEncoder` 最后对时间维 `mean(dim=0)`，把“B 何时刹车、C 何时响应、是否落在占用边界之前”压成一个静态 node embedding；排序真正需要的是事件窗口和 guard crossing。
4. 边由初始 `dx/dy` 启发式筛选，未验证完整可达 guard graph；它能做预测特征，不能作物理完备性证明。
5. `anchor_graph` 和 `anchor_full` 的关键排序结果表明，graph 的改善不能归因于 delayed graph；目前更像容量/正则化或 anchor feature 差异。
6. 代码没有把真实 total latency 代入占用安全余量，也没有在 latency 或 guard 不确定时强制 recompute。因此 delay 还是 feature，不是 validity condition。

建议的可检验结构是：

```text
T_total = T_encode + T_anchor + T_graph + T_calibration + T_fallback + T_rank
m_occ(t) = d(t) - q(v_close) T_total - 0.5 q(a_close) T_total^2 - e_guard
```

同时学习/估计 `K_ij(τ)=P(T_response,j-T_observation,i <= τ)`，而不是只用 mean delay：

- guard interval 不跨 0：允许 residual reuse；
- guard interval 跨 0：分支或候选条件 full recomputation；
- guard 缺失、policy family unsupported 或 latency 未测量：`ABSTAIN`。

新增的 [boundary_gate.py](sandbox:/workspace/scratch/2ddeb61865ff/work/code/caver/models/boundary_gate.py) 实现了这个保守决策接口，但它还不是物理 safety certificate。

## 5. 为什么显式刹车/事件特征没有提高排序

`full` 与 `no_events` 相同并不能证明刹车信息在原则上无用。当前实现存在四个可解释原因：

1. source pair contact/clearance 本身已经包含 braking dynamics；只屏蔽输入特征不等于移除 event-conditioned transition。
2. `pair_features` 把无效 event slack 存成 0，把无 onset 存成 9 秒；这混淆了“无事件、超出 horizon、censor、未检测”。
3. event/slack 主要通过 pointwise BCE、clearance 和低权重 slack loss 学习，最终候选排序没有独立的 event contrast objective。
4. `JointRiskHead` 从每个 pair feature 重复读取 candidate-level union anchor；pair 集合改变时，union representation 的统计含义也改变。

建议把 CBC 改成显式分解：

```text
event head:       P(event by t), onset hazard, q90 delay, event confidence
boundary head:    signed guard interval and stopping-slack interval
risk head:        pair contact and downstream union risk
contrast head:    Δhat(a,b,p) with shared potential, antisymmetry
```

训练目标使用：

```text
L = L_contact + λ_contrast Huber(Δhat, Δref)
  + λ_rank softplus(m - sign(Δref) Δhat)
  + λ_event BCE(event_by_t)
  + λ_onset censored-time loss
  + λ_slack Huber(slack_hat, slack_ref) on valid events only
```

其中 `L_rank` 只使用 reference CI 排除 0 且 `|Δref|>=δ` 的 pair；event missing 必须用 mask/censor indicator 表示。候选排序应直接读取 contrast head 或 upper-risk score，而不是由一个普通 MAE 最优的 head 间接决定。

## 6. v4 代码改动

### 扩展审计

- `caver/validation/extended_metrics.py`：critical sign/underresolution/miss 分离、scene-cluster bootstrap、pair MC CI、yield、explicit false-safe reuse、event timing、rare-risk calibration、runtime summary。
- `scripts/cbc_extended_audit.py`：无 torch 依赖的独立审计脚本，适合服务器上直接对 `.jsonl.gz` 运行。
- `caver/cli.py`：新增 `extended-audit` 子命令。
- `experiment.py`：为新生成的 records 加入 `runtime` transcript；明确标成离线 record-generation wall-clock，绝不伪称 online planning latency。
- `caver/models/boundary_gate.py`：把 measured latency 和 guard envelope 转成保守 `REUSE/RECOMPUTE/ABSTAIN` 决策。

### 测试

新增 `test_extended_metrics.py`、`test_cbc_extended_audit.py` 和 `test_boundary_gate.py`。当前容器没有 pytest/torch，因此只能做 Python 编译检查和无依赖 standalone audit 单元测试；服务器环境仍需运行完整 pytest。

## 7. 下一轮服务器执行顺序

### 第一步：解压 v4、安装环境、跑测试

```bash
unzip -o CaVER_CBC_v4_diagnostics.zip
cd CaVER_CBC_v4_diagnostics
python -m pip install -e '.[test]'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q
```

### 第二步：对已有记录做零重训扩展审计

原有 JSONL 没有 v4 runtime 字段，所以该命令会正确报告 runtime `NOT_MEASURED`，但可以马上得到 sign/underresolution、pair MC CI、scene-cluster flip CI、yield、rare-risk calibration：

```bash
python scripts/cbc_extended_audit.py \
  outputs/cbc_womd_study/test_records.jsonl.gz \
  --out outputs/cbc_womd_study/extended_audit.json \
  --bootstrap 2000

python scripts/cbc_extended_audit.py \
  outputs/cbc_synthetic_study/test_records.jsonl.gz \
  --out outputs/cbc_synthetic_study/extended_audit.json \
  --bootstrap 2000

# 如果要复核当前开发集冻结的经验 screen（clearance=24, guard=0），
# 用带 CaVER 依赖的 CLI；这仍然是 empirical screen，不是安全证书。
python -m caver.cli extended-audit \
  --input outputs/cbc_synthetic_study/test_records.jsonl.gz \
  --reuse-clearance-multiplier 24 \
  --reuse-guard-multiplier 0 \
  --output outputs/cbc_synthetic_study/extended_audit_with_screen.json
```

如果只上传报告包，不上传原始 JSONL：

```bash
python scripts/package_diagnostics.py \
  outputs/cbc_phase0_v3 outputs/gpu_ablation \
  --output outputs/CaVER_CBC_v4_upload_me.zip --max-mb 20
```

### 第三步：重新生成小规模记录以获得离线组件时延

必须重新 collect，旧记录无法事后恢复计时字段。先用合成数据确认 schema：

```bash
MODE=study RECORDS_ONLY=1 \
TRAIN_SCENES=40 TEST_SCENES=80 SOURCE_WORLDS=8 AUDIT_WORLDS=32 \
CBC_OUT=outputs/cbc_synthetic_v4 \
bash scripts/cbc_synthetic_study.sh 2>&1 | tee outputs/cbc_synthetic_v4_collect.log

python scripts/cbc_extended_audit.py \
  outputs/cbc_synthetic_v4/test_records.jsonl.gz \
  --out outputs/cbc_synthetic_v4/extended_audit.json \
  --bootstrap 2000
```

这些 `runtime` 数值是 record-collection wall-clock；要获得论文可用的真实 total planning latency，必须在真实 planning callback 入口/释放点再加一层 tracing，并记录 `callback_count`, `rollout_count`, `reused`, `fallback`, `deadline_miss`。

### 第四步：补齐可识别的 CBC 消融

必须同时跑：

```text
anchor_copy
anchor_full
anchor_graph
source
full
no_events
anchor_graph_equal_compute_no_graph
full_equal_compute_no_events
```

`anchor_graph` 只能与同一次 anchor source budget 的 non-graph control 比；`full` 只能与同容量、同 source budget 的 `no_events` 比。每组固定训练 scene、reference family、MC budget、epoch 和 random seeds，最后按 scene 做 cluster bootstrap。

### 第五步：WOMD 只先修 candidate yield

不要把 adaptive observation-only candidate 与 fixed candidate 混成一个实验。先报告：

```text
intent_matched
kinematic_feasible
legal_route (must be explicit)
both_ego_safe under reference
reference-resolved critical
```

当 `matched + legal + both-ego-safe` 仍为 0 时，停止扩大 CBC 训练，不进入 DSE/TIV。

### 第六步：再决定 DSE/TIV

只有当 CBC 在独立 response families 上满足：

- 非零且足够的 matched+legal+ego-safe+reference-resolved pair；
- strict sign error、underresolution、false-safe reuse 都有 scene-cluster CI；
- graph/event 与 equal-budget control 的增量比较通过；
- 真实 p95/p99 total latency 和 deadline miss 已测量；

才进入 DSE 的 risk-vs-coverage 和 TIV 的 query-vs-latency 闭环实验。当前数据不支持把 DSE/TIV 的“未拒绝”写成成功。

## 8. 交付文件

- [v4 代码与诊断包](sandbox:/workspace/scratch/2ddeb61865ff/CaVER_CBC_v4_diagnostics.zip)
- [v4 论文建议稿](sandbox:/workspace/scratch/2ddeb61865ff/work/interactive_planning_CBC_v4_recommended.tex)
- [v4 审计报告](sandbox:/workspace/scratch/2ddeb61865ff/work/CaVER_CBC_v4_audit_CN.md)
- [论文结构比较笔记](sandbox:/workspace/scratch/2ddeb61865ff/paper_story_notes.md)

