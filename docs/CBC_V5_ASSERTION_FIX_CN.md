# CBC v5 assertion 修复与附加审计（2026-11-05）

## 1. 原始失败的直接原因

`tests/test_cbc_v3_incremental_audit.py::test_novelty_requires_stronger_equal_budget_control` 的 fixture 声明：

- 2 个 reference families；
- 每个 family 96 个 worlds；
- 合计 `K=192` worlds/action。

但 `cbc_gate.py` 按该报告中的 `P=3`、`alpha=0.05`、critical contrast `delta=0.20` 计算：

\[
K_{\min}=\left\lceil\frac{2\log(2P/\alpha)}{\delta^2}\right\rceil=240.
\]

因此该 fixture 实际不满足 runbook 的 finite-MC resolution contract，gate 正确返回 `INSUFFICIENT_REFERENCE_EVIDENCE`。原测试期望 descriptive verdict，却使用了 under-powered fixture；这是测试数据与实验契约不一致，不是应当放宽生产 gate 的理由。

修复方式：
- 将“测试 weak anchor-copy 不足以证明 graph 增量”的 fixture 改为每个 family 128 worlds（总计 256），使它只测试 comparator/novelty 逻辑；
- 新增回归测试，明确验证 96 worlds/family（总计 192）即使有 40 个报告为 resolved 的 pair，仍会因预注册 MC 分辨率不足而 abstain；
- 新增 reference-world metadata mismatch 测试，避免 report 顶层 K 与 finite-MC 元数据不一致时被 gate 接受。

## 2. 额外发现并修复的问题

### 2.1 Primary critical ranking 曾未过滤 Ego-unsafe pairs

`cbc_study._score_prediction()` 之前只要求 `matched_kinematic_proxy`，就把 pair 加入 critical ranking；`ego_safe` 只用于 downstream regret。这与论文/研究协议中“同意图、匹配、两个候选均 reference-ego-safe”的 primary denominator 不一致。

现在：
- critical/sign/underresolution/miss 的主分母只包含 kinematic/intent/task-cost matched 且两个候选均 reference-ego-safe 的 pair；
- 显式 `lawful_route_match=False` 的 pair 排除；
- `NOT_EVALUATED`/`UNVERIFIED` 仅保留为 simulator-conditional 诊断，不计为 legal-verified；
- 分开输出 `matched_ego_safe_scenes`、`matched_legal_ego_safe_scenes`、`matched_ego_safe_legal_unknown_scenes`；
- 输出 reference-resolved critical pairs 覆盖的 scene 数，避免只看 pair count 而忽略 cluster 分布。

同时修正了训练端的 contrastive rank objective：此前它只检查 `matched_kinematic_proxy`，并用 `|delta|>=0.1` 激活排序项，没有检查 Ego-safe、intent/cost/feasibility 子条件，也没有排除 finite-MC unresolved contrasts。现在它要求 matched、intent/kinematic/task-cost 条件通过、两个候选 reference-ego-safe、route legality 不明确为 illegal，并且 pair 同时满足 `|delta_ref|>=0.20` 与 `|delta_ref|>r_MC`。Smooth-L1 contrast regression 和 margin ranking 都只作用于这个 resolved-critical mask。训练元数据记录 contrastive scene/pair 数量，equal-budget gate 也会比较这些数量和阈值。

注意：synthetic generator 目前将 route legality 标为 `NOT_EVALUATED`，因此仍然不能声称 matched+legal evidence。

### 2.2 Gate 对 component gain 的判定过于单一

之前 graph/event point advantage 与 paired CI 主要只看 sign error。现在 graph-vs-anchor-full、full-vs-no-events 的组件增益必须在以下四项上方向一致：

- reference-resolved sign error；
- all-critical underresolution；
- all-critical miss；
- matched ego-safe downstream decision regret。

只有四项点估计都改善，且四项 scene-cluster paired bootstrap CI 均支持改善，才会输出 preliminary component gain。弱 anchor-copy 对照仍只支持 descriptive simulator-conditional signal，不证明 graph/event 增量。

### 2.3 Bootstrap 有效重采样数未被 gate 检查

之前只要有 CI 两个端点就可能进入 component gain 判定，即使有效重采样数极少。现在每个 metric 记录：
- `resamples_requested`
- `resamples_valid`
- `resamples_valid_fraction`

Gate 至少要求 100 个有效重采样，且有效比例不低于 80%；否则该 CI 标记为 `NOT_MEASURED`。

### 2.4 训练轮数相同不代表 optimizer updates 相同，且 graph 参数预算更大

之前 `fit_cbc()` 的 dense variant 每个 epoch 会先做 pair mini-batch 更新，再做每个 candidate group 的 joint 更新；graph variant 则每个 candidate group 合并 pair/joint loss 做一次更新。因此即使 `EPOCHS=GRAPH_EPOCHS`，两者每个 epoch 的 optimizer updates 也不相等。另外 graph head 原宽度 24，active 参数量高于 anchor-full。

现在：
- 所有 variant 统一为每个 `(scene, action)` group 一次 pair+joint update；
- contrastive loss 只对 matched、Ego-safe 且至少含一个 finite-MC resolved critical pair 的 scene 启用，并并入该 scene 的 `late` action group 更新，不额外增加 optimizer step；
- graph width 调整为 16，active trainable parameters 与 anchor-full 相差小于 1%；
- graph 分支未使用的 dense pair head 冻结；
- checkpoint 保存 optimizer updates、active/total parameter count、seed、contrast weight 和 training scene-set digest；
- `cbc-study` 把这些元数据写入报告，`cbc-gate` 验证 graph-vs-anchor-full 和 full-vs-no-events 是否真正 equal-budget。

这比单纯比较 epochs 更严格。旧 checkpoint 缺少这些元数据时，gate 不会认可组件增益；需要用新代码重训 checkpoint。

### 2.5 正式 study 默认预算与 runbook 不一致

`cbc_synthetic_study.sh` 的 study 默认曾是：
- `AUDIT_WORLDS=96`/family；
- `CBC_EPOCHS=65`、`GRAPH_EPOCHS=20`；
- `BOOTSTRAP=1000`。

runbook 的正式协议要求每 family 128 worlds、anchor_full/anchor_graph 相同 optimizer updates、bootstrap 2000。现在默认值已对齐为 128/65/65/2000，并在正式训练时拒绝 graph/anchor epochs 不一致（除非显式标记 exploratory）。

`cbc_train_dual_gpu.sh` 的正式默认值现在是 `EPOCHS=65`、`GRAPH_EPOCHS=$EPOCHS`、`BOOTSTRAP=2000`、`CONTRAST_WEIGHT=0.10`；`cbc-study` CLI 默认 bootstrap 也统一为 2000。

### 2.6 Contrastive ablation 之前没有成对 gate

runbook 要求比较 `full_contrast` vs `full`、`anchor_graph_contrast` vs `anchor_graph`，但此前 `cbc-study` 只生成 graph-vs-anchor-full 和 full-vs-no-events 的 paired scene bootstrap。现在增加了两组 contrastive paired bootstrap，并在 `cbc-gate` 的 `contrastive_ablation` 中单独报告点估计、四项指标 CI 和 budget audit。Contrast weight 是被操纵的 treatment，因此允许 treatment 权重大于 control；但 optimizer updates、active parameters、seed、训练 scene set、resolved pair mask 和 contrast threshold 必须相同。contrastive loss 已合并到每个 scene 的 late-action update，不再通过额外 optimizer steps 获得更多训练预算。

## 3. 验证结果

- 完整 pytest：**83 passed**；
- `bash -n scripts/cbc_synthetic_study.sh scripts/cbc_train_dual_gpu.sh`：通过；
- `python -m py_compile` / `compileall`：关键修改模块通过；
- records-only synthetic smoke：通过；
- 1-epoch `anchor_full` / `anchor_graph` 训练 smoke：optimizer updates 均为 12（contrastive 与 non-contrastive 相同）；在 K=8 的 pilot 数据上 resolved-critical mask 为空，因此不启用 rank loss；active parameters 分别为 25,425 / 25,371（差异约 0.21%）；
- 旧 graph checkpoint（width=24）兼容加载 smoke：成功按 checkpoint 权重形状恢复 channels=24，并冻结未使用的 dense pair head；
- `cbc-study` + `cbc-gate` smoke：新 checkpoint 元数据成功传递，graph-vs-anchor-full、full-vs-no-events 及两组 contrastive budget audit 均能执行；contrastive treatment/control 在相同 updates 下返回 `MATCHED`；
- smoke `contract_audit.json`：`status=PASS`，train/calibration/test 分割和 seed overlap 检查无错误。

Smoke 使用小样本，只验证执行链路，不是算法性能证据。

## 4. 重要限制仍然存在

- `lawful_route_match` 目前在 synthetic records 中为 `NOT_EVALUATED`；
- ego-safe 仍是有限 reference MC 经验风险阈值判断，不是物理安全证书；
- offline collection timing 不能替代 online callback-to-release latency；
- empirical boundary screen 不是安全认证；
- WOMD observation history 不能识别未执行 Ego action 的反事实响应。

正式结论仍需遵守 `docs/CBC_V5_RUNBOOK_CN.md`。
