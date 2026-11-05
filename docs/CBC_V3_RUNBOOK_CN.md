# CaVER CBC v3：复现、服务器两卡训练及小体积结果回传（2026-11-05）

## 前提与输出规范

- 代码压缩包：`CaVER_CBC_v3_2026-11-05.zip`，内部根目录 `CaVER/`。
- 原始数据沿用既有目录，脚本不会自动下载大规模 WOMD：`/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/{training,validation}`。这阶段不用下载 `tf_example/training`。
- 已经准备好的 WOMD 压缩场景和结构仿真大记录**留在服务器**，不需上传。最终只回传 **报告白名单 ZIP**（默认小于20 MiB，硬上限小于50 MiB）。
- 数据来源声明：WOMD 仅提供**观测**真实初始状态和历史，`collect` 反事实结果是自定义仿真器的标签。
- 重要：第一阶段本质是诊断验证，不要把 `anchor_graph` 优势看成真实世界安全性能或安全证书。

## 1. 解压、确认依赖与 CUDA

```bash
unzip -o CaVER_CBC_v3_2026-11-05.zip
cd CaVER

# 强烈建议在你现有且已通过上一轮 pytest 的 Python 环境中进行：
python -m pip install -e '.[test]'

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m pytest -q
python - <<'PY'
import torch
print('PyTorch:',torch.__version__, 'CUDA available:',torch.cuda.is_available())
print('Device count:',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):print(i,torch.cuda.get_device_name(i))
PY
nvidia-smi
```

若 `torch.cuda.device_count()` 为 0/1 而 `nvidia-smi` 显示两张卡，需要检查 PyTorch 的 CUDA wheel、驱动和 `CUDA_VISIBLE_DEVICES`，不要盲目安装另一个 CUDA 或 TensorFlow 版本破坏 WOMD 环境。训练脚本在零 GPU 时安全回退 CPU；**在本地无 CUDA 环境下仅已验证 CPU 逻辑，未在你的机器实测两卡速度**。

## 2. 最优先执行：零重训的 Phase 0 分析（服务器上保留旧输出时）

先确认上一轮脚本产生的目录，通常是：

```bash
ls -lh outputs/cbc_synthetic_study/cbc_study.json
ls -lh outputs/cbc_womd_study/cbc_study.json
ls -lh outputs/cbc_womd_study/womd_{train,test}_corridor.jsonl.gz
```

如果存在，直接运行：

```bash
OUT=outputs/cbc_phase0_v3 \
WOMD=outputs/cbc_womd_study \
SYN=outputs/cbc_synthetic_study \
bash scripts/cbc_phase0_review.sh \
  2>&1 | tee outputs/cbc_phase0_v3_run.log

# 只上传这一份（小体积）
ls -lh outputs/cbc_phase0_v3/CaVER_CBC_v3_phase0_bundle.zip
```

它会生成：

- `synthetic_cbc_gate_v3.json`、`womd_cbc_gate_v3.json`：纠正后的证据门槛，包括 `novelty_verdict`。
- `train_womd_context_v3.json`、`test_womd_context_v3.json`：当前及历史的 BC 车距、TTC、跟驰 THW、急刹、制动裕度、地图保留情况。
- `train_candidate_design_v3.json`、`test_candidate_design_v3.json`：固定候选 vs 观测自适应候选的合法性之外的运动学/名义安全筛选通过率及失败理由。
- 测试摘要和 manifest。这一阶段无需生成 `cbc_study` 全量训练样本，也不会因 JSONL 过大无法上传。

`cbc_candidate_design_v3` 并没有使用 reference future 或碰撞标签，但“名义安全”是简单常速度几何检验、**不是**动态反应仿真的双候选安全，也不是地图合法性证明。

若缺少 `.jsonl.gz`，可以把第 3 节数据预处理重跑，但建议先核查旧目录；请勿上传原始 WOMD 观测 records。

## 3. Phase 1：重新生成有限的 WOMD 观测记录，优先检查候选可行性

若原始 `.jsonl.gz` 不存在，先运行：

```bash
# 只生成 observation 及训练/校准/测试记录，跳过耗时的 CBC 训练。
MODE=pilot RECORDS_ONLY=1 POLICY_DEVICE=auto \
CBC_OUT=outputs/cbc_womd_pilot_v3 \
bash scripts/cbc_womd_pipeline.sh \
  2>&1 | tee outputs/cbc_womd_pilot_v3.log

python scripts/inspect_womd_context.py \
  --input outputs/cbc_womd_pilot_v3/womd_test_corridor.jsonl.gz \
  --output outputs/cbc_womd_pilot_v3/test_womd_context_v3.json
python scripts/cbc_candidate_design.py \
  --input outputs/cbc_womd_pilot_v3/womd_test_corridor.jsonl.gz \
  --output outputs/cbc_womd_pilot_v3/test_candidate_design_v3.json
```

`womd-prep` 限制每个压缩文件最多45 MiB；一旦达到预算会停止预处理，并记录 manifest 的“未处理/跳过/样本不足”等信息，不会静默宣称扫描完整数据集。若 pilot 的有效匹配少，请先看原因，不要直接增加 100 倍推理预算。

若 Phase0 候选合格率值得继续，可设置研究规模先生成数据：

```bash
MODE=study RECORDS_ONLY=1 POLICY_DEVICE=cuda:0 \
POLICY_SCENES=3000 TRAIN_SCENES=120 CAL_SCENES=40 TEST_SCENES=120 \
MAX_SCAN=110000 MAX_SHARDS=380 PER_SHARD=550 \
SOURCE_WORLDS=8 AUDIT_WORLDS=96 \
CBC_OUT=outputs/cbc_womd_v3 \
bash scripts/cbc_womd_pipeline.sh \
  2>&1 | tee outputs/cbc_womd_v3_collect.log
```

上面会运行大量**CPU 结构响应仿真**，不等于 PyTorch 双 GPU 加速；`POLICY_DEVICE=cuda:0` 只影响 policy 训练阶段。reference worlds 是两种自定义合成响应族；增加抽样数量只能减少**此模拟器的 MC 方差**，不能纠正模型偏差。若 `test_candidate_design_v3.json` 的匹配+名义安全为零，应该停止这一步，回头修候选定义与地图筛选。

## 4. 双 GPU 做严格配对 CBC 消融（重点）

支持的独立变量：

- `full` vs `no_events`：同逐候选 source 预算，只测试显式事件特征的增量价值；源物理反应仍存在。
- `anchor_graph` vs `anchor_full`：同单 Anchor source 预算，只测试当前 graph head 的增量价值。
- `anchor_graph_contrast` vs `anchor_graph`：加入候选风险差监督是否改善临界排序。
- `full_contrast` vs `full`：在高预算 source 下同样检查 contrast loss 是否增益。

先选择实验规模：

```bash
# 小规模验证命令、模型保存与报告生成
EPOCHS=6 GRAPH_EPOCHS=3 BOOTSTRAP=100 CONTRAST_WEIGHT=0.3 \
bash scripts/cbc_train_dual_gpu.sh \
  outputs/cbc_womd_pilot_v3/train_records.jsonl.gz \
  outputs/cbc_womd_pilot_v3/test_records.jsonl.gz \
  outputs/cbc_womd_pilot_v3/gpu_ablation \
  outputs/cbc_womd_pilot_v3/calibration_records.jsonl.gz \
  2>&1 | tee outputs/cbc_womd_pilot_v3/gpu_ablation.log
```

如果 test 有足够关键样本才跑较大规模：

```bash
EPOCHS=65 GRAPH_EPOCHS=20 BOOTSTRAP=1000 CONTRAST_WEIGHT=0.3 \
bash scripts/cbc_train_dual_gpu.sh \
  outputs/cbc_womd_v3/train_records.jsonl.gz \
  outputs/cbc_womd_v3/test_records.jsonl.gz \
  outputs/cbc_womd_v3/gpu_ablation \
  outputs/cbc_womd_v3/calibration_records.jsonl.gz \
  2>&1 | tee outputs/cbc_womd_v3/gpu_ablation.log
```

训练脚本确认有两张 CUDA GPU 时自动使用两并行进程、每张卡一个作业；只发现一张/零张卡时顺序执行。`CUDA_VISIBLE_DEVICES` 为各子进程隔离卡，避免 GPU0/1 混用。训练权重不写到上传报告 ZIP（安全且避免大文件）；请在服务器保留 `.pt` 供后续对照。显存不足时下调两个脚本并行数、或单独 `CUDA_VISIBLE_DEVICES=0` 运行（此时识别为一张卡，顺序训练）。

## 5. 合成压力样本的配对消融（只作为算法调试）

```bash
MODE=study RECORDS_ONLY=1 \
TRAIN_SCENES=100 TEST_SCENES=120 SOURCE_WORLDS=8 AUDIT_WORLDS=96 \
CBC_OUT=outputs/cbc_synthetic_v3 \
bash scripts/cbc_synthetic_study.sh \
  2>&1 | tee outputs/cbc_synthetic_v3_collect.log

EPOCHS=65 GRAPH_EPOCHS=20 BOOTSTRAP=1000 CONTRAST_WEIGHT=0.3 \
bash scripts/cbc_train_dual_gpu.sh \
  outputs/cbc_synthetic_v3/train_records.jsonl.gz \
  outputs/cbc_synthetic_v3/test_records.jsonl.gz \
  outputs/cbc_synthetic_v3/gpu_ablation \
  2>&1 | tee outputs/cbc_synthetic_v3/gpu_ablation.log
```

缺失独立开发记录时，边界屏幕使用 training 作为开发数据，脚本会警告；虽然独立 test 未被使用作拟合，这种证据仍应标记为探索性。**合成实验的改善不能直接作为现实临界碰撞率的改善。**

## 6. 有限的 WOMD 自适应候选试验（不重新下载数据）

```bash
OLD=outputs/cbc_womd_study OUT=outputs/cbc_womd_adaptive_pilot \
TRAIN_SCENES=15 TEST_SCENES=15 SOURCE_WORLDS=3 AUDIT_WORLDS=10 \
bash scripts/cbc_womd_probe_v3.sh \
  2>&1 | tee outputs/cbc_womd_adaptive_pilot.log
```

该试验重用旧版 `policy_observed.pt`、旧版 WOMD 压缩 scenes，只需几份诊断即可判断新候选会不会继续把 120 个 WOMD 场景筛到近乎没有可用 pair；参考 MC=20/候选非常少，不可用于定量安全结论。先检查 `candidate_design_v3.json`，再决定是否花费更多推理预算。

## 7. 严格解读输出和科学决策门槛

重点依次打开：

```bash
python - <<'PY'
import json
from pathlib import Path
for p in (
  Path('outputs/cbc_phase0_v3/synthetic_cbc_gate_v3.json'),
  Path('outputs/cbc_phase0_v3/womd_cbc_gate_v3.json'),
  Path('outputs/cbc_womd_v3/gpu_ablation/cbc_gate.json')):
 if not p.exists():continue
 x=json.loads(p.read_text())
 print('\n',p, 'novelty_verdict:',x.get('novelty_verdict'))
 print('evidence_gaps:',x.get('evidence_gaps'))
 print('incremental_ablation:',x.get('incremental_ablation'))
 print('samples:',x.get('observed_sample'))
PY
```

解释规则：`INSUFFICIENT_REFERENCE_EVIDENCE` / `NOT_TESTABLE_WITH_CURRENT_EVIDENCE` 意味着不能判断 CBC 价值。`INCREMENTAL_GRAPH_AND_EVENT_GAIN_NOT_DEMONSTRATED` 表示当前对照没有显示机制增益，不等于断言自然界不存在级联。即使 `DESCRIPTIVE_COMPONENT_GAIN_NOT_STATISTICALLY_RESOLVED`，也只是趋势，不能写为证明。只有独立 holdout、合理场景簇 bootstrap、对照预算一致、足够临界 pair、多个真实反应模型族下重复改善，才进入更大规模实验。

额外建议的**有实际研究价值的指标**：critical sign error、critical underresolution/miss、false-safe anchor reuse、关键事件漏检和提前量、Ego 安全约束下的 downstream regret、reference pair flip 的组簇 MC CI、p95/p99 真正总延迟、每百次规划的 callback/rollout 数量、matched+legal+ego-safe pair yield。罕见风险还需分层 calibration，不能只用总体 AUC 或 MAE。

## 8. CARLA、Waymax、ReactSim 处于什么阶段？

```bash
# 合成四车挑战不调用 CARLA：不要用来测试 RPC。
python -m caver.cli fourcar-challenge --output outputs/fourcar_synthetic.json

# 你启动 CARLA0.9.16 的另一个终端后检查：
python -m caver.cli carla-check \
  --host 127.0.0.1 --port 2000 --timeout 10 \
  --output outputs/carla_check.json
```

Waymax 是官方基于 JAX 的 WOMD 轻量多代理仿真框架，可用 GPU；本代码保留了 Waymax 读取接口，**并没有实现完整 Waymax 官方闭环独立反应器的多个候选终端验证**。如需下一阶段正式研究，应先冻结候选和关键场景集，再分别运行不同 reactive policy 的闭环结果，统一导出源/参考审计格式。ReactSim-Bench 下载/预处理命令留在原项目 `scripts/download_reactsim.sh` 与 `scripts/reactsim_preprocess.sh`，同样**没有直接接入正式反应模型的推理训练**。本版不建议 DSE/TIV 联合训练。

## 9. 只回传这一个小 ZIP

```bash
python scripts/package_diagnostics.py \
  outputs/cbc_phase0_v3 \
  outputs/cbc_womd_v3/gpu_ablation \
  outputs/cbc_synthetic_v3/gpu_ablation \
  --output outputs/CaVER_CBC_v3_upload_me.zip \
  --max-mb 20 \
  --log outputs/cbc_womd_v3/gpu_ablation.log

ls -lh outputs/CaVER_CBC_v3_upload_me.zip
```

若尚未执行某个 stage，打包器会把不存在的目录列入 `excluded`，并打包已经存在的报告；要求至少有一个有效报告。仅上传 `outputs/CaVER_CBC_v3_upload_me.zip`；不要上传原始 `.jsonl.gz`、checkpoint 或 WOMD Shards。上传后才能对 **当前历史制动/TTC 的分布、候选选择是否改善、双卡训练的真实结果** 给出结论。

## 10. 与最新论文文件的对应关系

论文的修改版 `interactive_planning_CBC_evidence_revised_2026-11-05.tex` 保留 CBC/DSE/TIV 原始结构、方程和参考文献键，新增：实测 v2 报告表格、描述性和增量因果证据拆分、同预算图/事件消融要求、可选训练 pairwise loss、工程限制。已通过 `pdflatex -draftmode -halt-on-error` 语法检查；这次未上传原稿引用的 `interactive_planning.bib`，因此缺少引用解析，要在你原论文环境中合并文献文件并运行 `biber` 后编译完整稿件。
