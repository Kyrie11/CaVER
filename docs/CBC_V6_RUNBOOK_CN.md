# CBC v6 实验执行说明

本版本先修复统计契约和图条件 joint head，再验证两个候选机制。所有变体必须使用同一组 train/calibration/test records、seed、epoch、optimizer update 数和 reference world 预算。

## 1. 安装与静态检查

```bash
set -euo pipefail
python -m pip install -e '.[test]'
python -m pytest -q
python -m compileall -q caver scripts
```

## 2. 生成正式 records

```bash
OUT=outputs/cbc_synthetic_v6
rm -rf "$OUT"
mkdir -p "$OUT"

MODE=study \
CAL_SCENES=40 TRAIN_SCENES=100 TEST_SCENES=240 \
SOURCE_WORLDS=8 AUDIT_WORLDS=128 RECORDS_ONLY=1 \
CBC_OUT="$OUT" bash scripts/cbc_synthetic_study.sh \
  2>&1 | tee "$OUT/collect.log"

cat "$OUT/manifest.json"
cat "$OUT/contract_audit.json"
```

`contract_audit.json` 必须是 `PASS`。每个 reference family 使用 128 worlds，总计 256 worlds/action。`route legality` 在当前 synthetic protocol 仍是 `UNVERIFIED`，不得把 matched yield 写成 matched+legal yield。

测试集扩展到 240 个 scene 是为了在 v5 约 17.5\% 的 resolved-critical 产率下获得至少 30 个严格 resolved critical pairs。若产率下降，继续增加测试 scene，不能通过降低 critical threshold 或 finite-MC 分辨率标准来凑足门槛。

## 3. 固定训练预算

```bash
SEED=11 EPOCHS=65 GRAPH_EPOCHS=65 BOOTSTRAP=2000 CONTRAST_WEIGHT=0.10 \
bash scripts/cbc_train_dual_gpu.sh \
  "$OUT/train_records.jsonl.gz" \
  "$OUT/test_records.jsonl.gz" \
  "$OUT/gpu_ablation" \
  "$OUT/calibration_records.jsonl.gz" \
  2>&1 | tee "$OUT/train.log"
```

v6 代码新增了图条件 downstream joint head。旧 v5 checkpoint 可以加载，但新实验必须重新训练。`anchor_graph` 与 `anchor_full` 的 optimizer updates、active parameter count 和 seed 必须一致。

## 4. 训练 delay-kernel 与事件局部池化变体

```bash
python -m caver.cli train-cbc \
  --input "$OUT/train_records.jsonl.gz" \
  --variant anchor_graph_delay_kernel \
  --epochs 65 --seed 11 --device auto \
  --output "$OUT/gpu_ablation/cbc_anchor_graph_delay_kernel.pt"

python -m caver.cli train-cbc \
  --input "$OUT/train_records.jsonl.gz" \
  --variant anchor_graph_event_local \
  --epochs 65 --seed 11 --device auto \
  --output "$OUT/gpu_ablation/cbc_anchor_graph_event_local.pt"
```

`anchor_graph_delay_kernel` 使用 source policy family 的 response-delay q10/q50/q90 构造离散 arrival kernel。`anchor_graph_event_local` 只在最后三个时间片做事件加权池化。两者是机制消融，不能直接替代主模型。

## 5. held-out CBC study

```bash
python -m caver.cli cbc-study \
  --input "$OUT/test_records.jsonl.gz" \
  --model full="$OUT/gpu_ablation/cbc_full.pt" \
  --model no_events="$OUT/gpu_ablation/cbc_no_events.pt" \
  --model anchor_full="$OUT/gpu_ablation/cbc_anchor_full.pt" \
  --model anchor_graph="$OUT/gpu_ablation/cbc_anchor_graph.pt" \
  --model anchor_graph_event_local="$OUT/gpu_ablation/cbc_anchor_graph_event_local.pt" \
  --model anchor_graph_delay_kernel="$OUT/gpu_ablation/cbc_anchor_graph_delay_kernel.pt" \
  --contrast .20 --detect .05 --bootstrap 2000 \
  --output "$OUT/gpu_ablation/cbc_study_v6.json"

python -m caver.cli cbc-gate \
  --study "$OUT/gpu_ablation/cbc_study_v6.json" \
  --boundary "$OUT/gpu_ablation/cbc_boundary_study.json" \
  --min-resolved 30 --min-matched 50 --min-scenes 100 \
  --output "$OUT/gpu_ablation/cbc_gate_v6.json"
```

`cbc_study_v6.json` 中 `finite_mc_radius.n_worlds` 必须等于 256，`n_pairs` 必须等于场景内 pair 数。若仍出现 `FINITE_MC_WORLD_COUNT_METADATA_MISMATCH`，停止解释结果并检查使用的代码版本。

## 6. extended audit

```bash
for name in source full no_events anchor_full anchor_graph \
  anchor_graph_event_local anchor_graph_delay_kernel; do
  if [[ "$name" == source ]]; then
    python -m caver.cli extended-audit \
      --input "$OUT/test_records.jsonl.gz" --bootstrap 2000 \
      --output "$OUT/extended_${name}.json"
  else
    python -m caver.cli extended-audit \
      --input "$OUT/test_records.jsonl.gz" \
      --checkpoint "$OUT/gpu_ablation/cbc_${name}.pt" \
      --bootstrap 2000 \
      --output "$OUT/extended_${name}.json"
  fi
done
```

新 records 会保存 right-censor-aware `event_timing_labels`。extended audit 应报告 event timing 的 recall、presence、onset error 和 lead-time。没有显式在线 `reuse_decisions` 时，false-safe anchor reuse 仍保持 `NOT_MEASURED`。

## 7. 重复与报告规则

至少用 `SEED=11,29,47` 重复全部模型和同一 records。主结论使用 scene-cluster bootstrap。只有在 reference-resolved critical pairs 达到预注册门槛、graph 与 no-graph 同预算、事件 replacement 同容量、以及 runtime 具备真实 callback-to-release trace 时，才可写 CBC 组件的独立增益。rollout count reduction 只能作为计算量 proxy，不能写成真实 planner latency speedup。

## 8. CBC 冻结后再评估 DSE 与 TIV

先在 calibration split 上拟合 whole-episode reference-mean calibration。校准文件必须记录 checkpoint digest、reference family mix 和 query/hold procedure。

```bash
python -m caver.cli calibrate \
  --input "$OUT/calibration_records.jsonl.gz" \
  --checkpoint "$OUT/gpu_ablation/cbc_anchor_graph.pt" \
  --output "$OUT/calibration_anchor_graph.json"

python -m caver.cli evaluate \
  --input "$OUT/test_records.jsonl.gz" \
  --checkpoint "$OUT/gpu_ablation/cbc_anchor_graph.pt" \
  --calibration "$OUT/calibration_anchor_graph.json" \
  --eps-ego .08 --eps-down .12 --eps-inc .10 --eps-reg 1.5 \
  --output "$OUT/dse_anchor_graph.json"
```

DSE 比较必须同时报告 risk-admissible coverage、decision-certified coverage、abstention、accepted-action regret、joint interval coverage 和 LP witness 类型。没有 verified holding prefix 时，`DECISION-CERTIFIED` 不得释放。

TIV acquisition head 使用 train records 学习，使用独立 calibration records 审计。v6 的 `--validation` 参数避免把同一批 query labels 同时用于训练和审计。

```bash
python -m caver.cli tiv-audit \
  --input "$OUT/train_records.jsonl.gz" \
  --validation "$OUT/calibration_records.jsonl.gz" \
  --head "$OUT/tiv_value_head.pt" \
  --epochs 120 --budget 12 \
  --output "$OUT/tiv_validation.jsonl"
```

TIV 的 signed value 仍是 simulator-conditional acquisition proxy。必须单独记录 query latency、holding-prefix validity、query source、post-query recalibration 和 terminal DSE outcome。TIV 不能因为预测 value 为正而替代 DSE，也不能使用 terminal calibration split 决定下一次 query。
