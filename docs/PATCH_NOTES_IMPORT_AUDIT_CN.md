# CaVER CBC v4 补丁说明：pytest 导入路径与论文指标对齐

## 1. 报错原因

`tests/test_cbc_extended_audit.py` 使用：

```python
from cbc_extended_audit import (...)
```

但模块文件实际位于 `scripts/cbc_extended_audit.py`。原 `pyproject.toml` 只配置了 `testpaths = ["tests"]`，没有把 `scripts/` 加入 pytest 的模块搜索路径。因此 `python -m pytest -q` 在收集测试时无法解析顶层模块名，尚未运行任何 CBC 测试或实验代码。

## 2. 修复内容

- `pyproject.toml`：在 pytest 配置中加入 `pythonpath = ["scripts"]`。这与项目声明的 `pytest>=7` 兼容，使测试导入路径与实际脚本位置一致。
- `scripts/cbc_extended_audit.py`：
  - 让 `critical_pair_audit(alpha=...)` 真正把 alpha 传入有限 Monte Carlo 半径计算；
  - 按所选 `early/late` 候选名检查匹配记录；
  - 按论文 Evaluation Protocol 的 Metric definitions，将**reference-resolved critical pairs**作为 sign error、underresolution 和 critical miss 的主要分母；
  - 新增 `critical_miss = sign_error OR underresolution`；
  - 保留全体 critical pairs 的 underresolution/miss 作为补充诊断，分别命名为 `underresolution_rate_all_critical` 和 `critical_miss_rate_all_critical`；
  - 对主要指标加入 scene-cluster bootstrap，整场景重采样，不把同一场景中的多个 pair 当成独立样本；
  - `--bootstrap` 同时控制 critical-pair 和 reference-flip 的 scene bootstrap；新增可复现的 `--seed`。
- `tests/test_cbc_extended_audit.py`：新增回归测试，覆盖 reference-resolved 分母和 scene-cluster bootstrap。

## 3. 与论文的关系

以上指标调整依据论文 `Evaluation Protocol and Falsification Criteria` 下的 `Metric definitions and denominators`：
- critical pair：`|Delta_ref| >= delta`；
- reference-resolved：`|Delta_ref| > r_MC`；
- sign error 与 underresolution 分开；
- critical miss 是两者的并集；
- 主要指标分母为 reference-resolved critical set；
- 置信区间按 scene cluster bootstrap。

本补丁**没有**修改 CBC 模型结构、训练目标、数据生成、候选动作、source/reference reactor、实验阈值或论文公式；也不需要修改上传的 TeX。它只修复测试导入配置，并使独立审计脚本的主要 critical-pair 统计口径符合论文已有定义。

## 4. 验证

在补丁工作副本中完成：
- `python -m pytest -q tests/test_cbc_extended_audit.py`：4 passed；
- `python -m pytest -q`：69 passed；
- `MODE=study RECORDS_ONLY=1` 的小规模 smoke test（1 train scene、1 test scene、1 source world、2 audit worlds）：脚本通过测试、four-car challenge 和 train/test record collection；
- 对 smoke-test 生成的 `.jsonl.gz` 执行 `scripts/cbc_extended_audit.py --bootstrap 2000`：CLI 正常运行。

小规模 smoke test 只验证执行链路，不代表正式 40/80-scene 研究结果。
