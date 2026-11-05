# CaVER/CBC v5 重跑说明

本版本先修复实验契约和指标定义，再评估算法组件。当前上传的 `cbc_synthetic_v4.zip` 实际是 train=40、test=80、每个候选 64 个 reference worlds 的小规模 runtime schema 试验，与用户声明的 100/120/96 命令不一致；不要把它当作第四步 study gate。

## 1. 安装和测试

```bash
rm -rf CaVER_CBC_v5_contract_patch
mkdir CaVER_CBC_v5_contract_patch
unzip -q CaVER_CBC_v5_contract_patch.zip -d CaVER_CBC_v5_contract_patch
cd CaVER_CBC_v5_contract_patch
python -m pip install -e '.[test]'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -q
```

必须保存 pytest 输出；任何失败都先修复，不进入 study。

## 2. 先做小规模 smoke

```bash
set -o pipefail
rm -rf outputs/cbc_synthetic_v5_smoke
MODE=pilot CAL_SCENES=2 TRAIN_SCENES=4 TEST_SCENES=4 \
SOURCE_WORLDS=4 AUDIT_WORLDS=8 RECORDS_ONLY=1 \
CBC_OUT=outputs/cbc_synthetic_v5_smoke \
bash scripts/cbc_synthetic_study.sh 2>&1 | tee outputs/cbc_synthetic_v5_smoke.log
cat outputs/cbc_synthetic_v5_smoke/manifest.json
cat outputs/cbc_synthetic_v5_smoke/contract_audit.json
```

`contract_audit.json` 必须是 `PASS`，并且 train/calibration/test 的 scene ID 和 synthetic seed 不得重叠。

## 3. 正式证据记录

使用两种 stress reference family，每个 family 128 个世界，即每个 action 总计 256 个 paired worlds。按照

\[
r_{MC}=\sqrt{2\log(2P/\alpha)/K},\quad P=3,\;\alpha=.05,
\]

256 个世界将分辨率半径压到约 0.193，能够基本把 0.20 的 critical threshold 与 MC 误差区分开。若计算预算不足，最低应使用每 family 120 个世界（总计 240，半径约 0.200）；`AUDIT_WORLDS=96` 只产生 192 个世界，半径约 0.223，仍不能达到 0.20 的预注册分辨率目标。

```bash
set -euo pipefail
set -o pipefail
OUT=outputs/cbc_synthetic_v5
rm -rf "$OUT"
mkdir -p "$OUT"
MODE=study CAL_SCENES=40 TRAIN_SCENES=100 TEST_SCENES=120 \
SOURCE_WORLDS=8 AUDIT_WORLDS=128 RECORDS_ONLY=1 \
CBC_OUT="$OUT" \
bash scripts/cbc_synthetic_study.sh 2>&1 | tee "$OUT/collect.log"
cat "$OUT/manifest.json"
cat "$OUT/contract_audit.json"
```

如果 `contract_audit.json` 不是 `PASS`，不要训练，也不要手工删除错误记录。

## 4. 公平的 CBC 消融训练

`anchor_full` 与 `anchor_graph` 不只要求相同 epochs：当前训练循环对所有变体统一为每个 `(scene, action)` group 一次 optimizer update，contrastive ablation 也对每个 matched scene 统一增加一次 update。因此相同 records、seed、epochs 和 contrast weight 下，optimizer update 数完全一致。`anchor_graph` 的 graph width 固定为 16，其 active trainable parameter count 与 `anchor_full` 的差异小于 1%；graph 分支未使用的 dense pair head 被冻结，不计入 active capacity。`cbc-study` 会把训练元数据写入报告，`cbc-gate` 会检查 update 数、active parameter count、seed、contrast weight 和训练场景集合；元数据缺失或不匹配时，不允许形成组件增益结论。

脚本仍默认拒绝 `EPOCHS != GRAPH_EPOCHS`，除非显式设置 `ALLOW_UNMATCHED_EPOCHS=1` 做 exploratory run。正式命令使用相同的 65 epochs：

```bash
SEED=11 EPOCHS=65 GRAPH_EPOCHS=65 BOOTSTRAP=2000 CONTRAST_WEIGHT=0.10 \
bash scripts/cbc_train_dual_gpu.sh \
  "$OUT/train_records.jsonl.gz" \
  "$OUT/test_records.jsonl.gz" \
  "$OUT/gpu_ablation" \
  "$OUT/calibration_records.jsonl.gz" \
  2>&1 | tee "$OUT/train.log"
```

正式论文结果至少重复 `SEED=11,29,47`；三次运行必须使用相同的 records、epochs、world budget 和 checkpoint 评估脚本，再按 scene cluster 汇总，而不能只挑最好的一次。

旧版 v5 checkpoint 若没有 `optimizer_updates_total`、`active_parameter_count`、训练场景集合摘要等元数据，会被 `cbc-gate` 标记为 equal-budget `NOT_VERIFIED`。为形成 graph/event 组件结论，应使用当前代码重新训练 checkpoint；旧 records 可以继续使用，只要其 train/calibration/test 契约审计通过。

在主消融通过契约检查后，再训练事件窗口池化的探索性图变体。它保持 `GraphCascadeRiskHead` 的参数形状不变，只将 `h.mean(dim=0)` 替换为由 source brake-guard 强度加权的时间池化；这一步用于检验“边界被全时段平均抹掉”这一机制假设，不能与主结果混在一起：

```bash
python -m caver.cli train-cbc --input "$OUT/train_records.jsonl.gz" \
  --variant anchor_graph_event --epochs 65 --device auto \
  --output "$OUT/gpu_ablation/cbc_anchor_graph_event.pt"
python -m caver.cli cbc-study --input "$OUT/test_records.jsonl.gz" \
  --model anchor_graph_event="$OUT/gpu_ablation/cbc_anchor_graph_event.pt" \
  --model anchor_full="$OUT/gpu_ablation/cbc_anchor_full.pt" \
  --model anchor_graph="$OUT/gpu_ablation/cbc_anchor_graph.pt" \
  --contrast .20 --detect .05 --bootstrap 2000 \
  --output "$OUT/gpu_ablation/cbc_study_event_pool.json"
```

第四个参数是独立 calibration split。它用于拟合 empirical boundary screener；若省略，脚本会把 train 当 development，并明确标记 exploratory。

## 5. 扩展审计

优先使用 CLI 版本，它读取新记录的 `runtime` 字段并排除 attribution-only `reference` action 的 calibration 点：

```bash
for name in source full no_events anchor_full anchor_graph anchor_graph_contrast full_contrast; do
  case "$name" in
    source) : ;;
    *) ck="--checkpoint $OUT/gpu_ablation/cbc_${name}.pt" ;;
  esac
  if [[ "$name" == source ]]; then
    python -m caver.cli extended-audit --input "$OUT/test_records.jsonl.gz" \
      --bootstrap 2000 --output "$OUT/extended_${name}.json"
  else
    python -m caver.cli extended-audit --input "$OUT/test_records.jsonl.gz" \
      --checkpoint "$OUT/gpu_ablation/cbc_${name}.pt" \
      --bootstrap 2000 --output "$OUT/extended_${name}.json"
  fi
done
```

`anchor_copy` 没有独立 checkpoint，历史记录也没有把它写入 `predicted`；因此不能把 `extended-audit` 的默认预测误称为 anchor-copy。它由 `cbc-study` 的成对比较报告，或在下一版记录中显式保存 `predicted_anchor_copy` 后再单独审计。

注意：当前记录没有 legal label、explicit event trigger 和在线 callback trace，因此对应字段必须保持 `NOT_MEASURED`。`runtime` 的 p95/p99 只描述离线 record collection；它不能被写成在线规划延迟。

## 6. 结果判定顺序

当前代码的 study 默认值已与本运行手册对齐：`AUDIT_WORLDS=128`（每个 reference family），`CBC_EPOCHS=65`、`GRAPH_EPOCHS=65`，bootstrap 默认 2000。若显式把图模型训练轮数设成与 anchor-full 不同，正式 study 会拒绝继续；只有设置 `ALLOW_UNMATCHED_EPOCHS=1` 才允许做探索性运行。

主 critical-ranking 分母现在严格限制为：
- 两个候选通过预先定义的 kinematic/intent/task-cost matching；
- 两个候选在 reference 下都满足 Ego-risk 阈值；
- 若 route legality 明确为 `False`，该 pair 排除；
- `lawful_route_match=NOT_EVALUATED/UNVERIFIED` 仅可进入 simulator-conditional 诊断，绝不计为 legal-verified。

因此 `matched_ego_safe_scenes` 与 `matched_legal_ego_safe_scenes` 分开输出。合成协议目前将 route legality 标为 `NOT_EVALUATED`，不能把它写成 matched+legal 证据。回归 MAE 仍是更广泛的候选风险拟合诊断，不等同于上述 primary ranking 分母。

contrastive rank objective 采用同一候选契约，并只对 finite-MC resolved critical pair 启用；它合并到对应 scene 的 `late` action group 更新，不额外增加 optimizer step：`|delta_ref| >= 0.20` 且 `|delta_ref| > r_MC`。未匹配、Ego-unsafe、明确 illegal 或 MC-unresolved 的 pair 不参与 contrastive Smooth-L1/ranking loss；这些场景仍可参与普通候选级监督，但不能提供关键排序标签。

paired scene bootstrap 除了输出 CI，还会输出 `resamples_requested`、`resamples_valid` 和有效比例。Gate 只有在至少 100 个有效 resamples 且有效比例不低于 80% 时，才把该 CI 作为 component-gain 支持；有效样本不足时返回 `NOT_MEASURED`，不把少数非空重采样形成的窄区间当成证据。

1. 先看 `contract_audit.json`。
2. 再看 `cbc_study.json` 中统一的 `finite_mc_radius`、纯 `sign_error`、`underresolution`、`critical_miss`。
3. 只在 reference-resolved critical pair 数达到预注册下限时解释 primary ranking；否则 verdict 保持 `NOT_TESTABLE_WITH_CURRENT_EVIDENCE`。
4. `full` 对 `no_events` 只报告匹配容量下的增量；事件只在 critical sign/underresolution 和 matched decision regret 同时改善且 scene-cluster CI 排除 0 时才能写成排序贡献。
5. `anchor_graph` 必须同时与 `anchor_copy` 和相同训练步数、参数规模匹配的 `anchor_full` 比较。当前实现还没有 graph-conditioned joint downstream head，因此 graph 的 union-risk 结果只能称为 joint-head/projection 输出，不能称为 graph 直接改善了 downstream union。
6. boundary study 只能称为 fitted empirical screener；缺少完整 guard envelope、legal route 和在线 latency 时不得称为安全证书。


### Contrastive loss ablation

`cbc-study` 会额外生成 `full_contrast` vs `full`、`anchor_graph_contrast` vs `anchor_graph` 的 paired scene-cluster bootstrap；`cbc-gate` 在 `contrastive_ablation` 单独报告两组结果。Contrast weight 是该消融的 treatment，因此允许 contrastive checkpoint 的权重大于 control；其余 optimizer updates、active parameter count、seed、训练场景集合、resolved-critical pair mask 和阈值必须一致。该对照不替代 graph/event 的独立消融。
