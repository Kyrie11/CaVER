# CBC v6 event-timing assertion 修复说明

## 1. 根因

失败断言是：

```python
assert out['event_timing']['status'] == 'NOT_MEASURED'
```

测试和 `caver/validation/extended_metrics.py` 的诊断逻辑均未出错。问题来自 `caver/experiment.py` 的 record producer：`paired_scene()` 把 `_event_timing_labels()` 的输出写入 `event_timing_labels`。

该函数从 source/reference simulator rollout 的 `response_onset` 计算：

- source reactor 中观测到的响应起始时间；
- held-out reference reactor 中观测到的响应起始时间；
- 对应的 onset quantile 和 censor 标志。

旧字段却将 source rollout onset 命名为 `predicted_trigger_s`。这会让 `event_timing_summary()` 将离线 source-response onset 误认成 CBC detector 的预测触发时间。它不是 CBC 在线/算法 detector 的触发时间，因此不能据此计算 CBC event recall、precision 或 lead time。

## 2. 修复原则

严格遵守“不修改诊断逻辑和测试逻辑”：

- `tests/test_extended_metrics.py` 未修改；
- `caver/validation/extended_metrics.py` 未修改；
- `event_timing_summary()` 仍只消费 `event_timing_labels` 或 `events`；
- record producer 将离线 source/reference onset 信息改存到独立命名空间 `offline_response_onset_diagnostics`。

字段名同时改为准确表达来源：

- `source_response_present`；
- `source_response_onset_s`；
- `source_response_onset_q10_s/q90_s`；
- `source_response_censored`；
- `reference_response_present`；
- `reference_onset_s`；
- `reference_onset_q10_s/q90_s`；
- `reference_censored`。

并声明：

```text
OFFLINE_SOURCE_VS_REFERENCE_RESPONSE_ONSET_NOT_CBC_TRIGGER
```

`docs/CBC_V6_RUNBOOK_CN.md` 同步修正，不再声称这些 onset summaries 足以测量 detector recall/lead-time。

## 3. 修复后正确的指标状态

在没有显式 CBC event-trigger transcript 时：

```json
{
  "event_timing": {
    "status": "NOT_MEASURED"
  }
}
```

这不是实验失败，而是正确的证据状态。其它可测量指标仍正常输出，例如：

- critical sign error / underresolution；
- reference pair flips；
- runtime collection statistics；
- rare-risk calibration；
- candidate yield。

其中 runtime 仍然是 offline record-collection wall-clock，不能当成在线 planner callback latency。

若未来要把 `event_timing` 变为 `MEASURED`，必须在 CBC 推理时真实记录 detector trigger timestamp，并与冻结的 held-out reference event label 按 scene/action/actor 对齐；不能再用 source rollout 的实际 onset 代替。

## 4. 变更文件

仅修改：

- `caver/experiment.py`
- `docs/CBC_V6_RUNBOOK_CN.md`

没有修改：

- `caver/validation/extended_metrics.py`
- `tests/test_extended_metrics.py`

## 5. 验证

本修复版本验证结果：

```text
python -m pytest -q
83 passed

python -m compileall -q caver scripts
PASS

bash -n scripts/*.sh
PASS
```

另运行 records-only pilot：

- train=4 scenes；
- calibration=2 scenes；
- test=4 scenes；
- source worlds=2；
- audit worlds/family=2。

结果：

```text
contract_audit.status = PASS
event_timing.status = NOT_MEASURED
runtime.status = MEASURED
runtime.scope = OFFLINE_RECORD_COLLECTION_WALLCLOCK_NOT_ONLINE_PLANNING_LATENCY
```

正式 100/40/240 scenes、8 source worlds、128 audit worlds/family 的完整 study 未在本修复环境中运行；需在实验服务器执行。

## 6. 服务器应用方式

在代码根目录解压覆盖：

```bash
unzip -o CaVER_v6_event_timing_fix.zip -d .
python -m pytest -q
python -m compileall -q caver scripts
for f in scripts/*.sh; do bash -n "$f"; done
```

然后重新生成 records。旧 records 中仍保留旧字段 `event_timing_labels`，不应与修复后的 records 混合；请使用全新的输出目录。

```bash
OUT=outputs/cbc_synthetic_v6_fixed
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

脚本会先运行 pytest；只有全部测试通过才会继续 collection。Records 生成后，contract audit 必须为 `PASS`。
