# CaVER v2：CBC 首先证伪实验、WOMD 小体积数据、CARLA 0.9.16 故障排查与复现实验手册

> 日期：2026-11-05。依据上传的最新 `interactive planning(2).tex`、上一版 CaVER ZIP 与 `result.zip` 进行修订。本工程仍属 **model-conditional research harness**，并非论文所有数学假设的完成认证。不要宣称 WOMD 单条日志证明了反事实人类驾驶结果。

## 0. 实际审计发现与实验定位

从用户 `result.zip` 能确认：

- `fourcar_challenge.json` **已经成功运行**，Early 的 B–C clearance 为约 **−1.800 m**（碰撞），Late 为 **+1.509 m**（不碰撞）；Ego 均未接触。此命令调用 Python 自带延迟 IDM **合成**响应器，**不使用 CARLA**。报告中 `no CARLA physics` 不是 `import carla` 报错。
- WOMD `scenario/training` 文件 1000 分片、`scenario/validation` 文件 150 分片，双方已成功解码 3 个场景；`tf_example/validation` 虽有分片但未证明解码；`tf_example/training` 未安装且 CBC 当前**不需要**。
- `doctor.json` 检测到了 CARLA Python 模块，但不代表 TCP RPC 2000 能连上。要用本版 `carla-check` 真正检查。
- `result.zip` 没有 CBC 模型/测试指标，故原上传数据**不能**判断 CBC 有效。

**三条完全不同的证据链：**

1. `fourcar-challenge`：冻结的确定性合成反例和媒介干预，是可实现性/机制检查。
2. `cbc_synthetic_study` 与 `cbc_womd_pipeline`：不同**代理反应模型**之间的候选响应排序、事件边界、单-anchor reuse 研究；后者源初始状态和历史来自 WOMD，但未来的备选轨迹由合成响应器生成，不能称为 WOMD 观察到的潜在结果。
3. `carla-run` / 后续 Waymax/ReactSim：外部闭环/物理模型的独立审计入口。目前的 CARLA runner 是脚本车辆控制的探索性试验，不是 WOMD/nuPlan 官方排行榜复现。

## 1. 一次安装与参数检查

建议使用 Python 3.10 单独环境。TensorFlow/WOMD wheel 需要在服务器匹配本地平台和 Waymo 开发包的 Python/TF 版本（不要为了安装 TF 覆盖已工作的 PyTorch/CARLA 环境）。

```bash
unzip CaVER_CBC_revised_2026-11-05.zip
cd CaVER
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
# 已有 WOMD Scenario 成功读取环境时，沿用既有、兼容的 TF/WOD 安装；无需重新强制安装：
python -c 'import tensorflow as tf; from waymo_open_dataset.protos import scenario_pb2; print(tf.__version__)'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
python -m caver.cli doctor --output outputs/doctor.json
bash scripts/verify_womd.sh
```

如果没有 `python3.10`，可在已完成依赖兼容性检查的 Python 3.10/3.11 环境运行，但部分 CARLA 0.9.16 wheels 和 nuPlan 官方 devkit 对解释器版本敏感，务必分别校验。

配置在 `configs/paths.json`：

```text
WOMD   /data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/
       uncompressed/scenario/{training,validation}/
nuPlan /data0/senzeyu2/dataset/nuplan/data/cache/public_set_*/   (*.db)
Maps   /data0/senzeyu2/dataset/nuplan/maps/
CARLA  127.0.0.1:2000
ReactSim /data0/senzeyu2/dataset/ReactSim-Bench/
```

若路径不同，可运行 `python -m caver.cli --paths /绝对路径/paths.json ...`，其中 `--paths` 在**子命令之前**。

## 2. CARLA：先查 RPC，再跑独立物理四车

**先说明错误来源**：

```bash
python -m caver.cli fourcar-challenge --output outputs/fourcar_challenge.json
```

不连接 CARLA，也无需启动 UE4。`no CARLA physics` 是防止错报物理证据的说明。

如果要检查实际 CARLA 服务器，在**终端 A** 中（到你安装的 CARLA 0.9.16 根目录）：

```bash
./CarlaUE4.sh /Game/Carla/Maps/Town04 -RenderOffScreen -carla-rpc-port=2000
```

在另一个 **终端 B** 中，进入 CaVER 虚拟环境：

```bash
cd /你的路径/CaVER
source .venv/bin/activate
python -c 'import carla;print(carla.__file__)'
python -m caver.cli carla-check --host 127.0.0.1 --port 2000 --timeout 10 --output outputs/carla_check.json
cat outputs/carla_check.json
```

读取 `rpc_connected`、`version_match`、`server_version`、`current_map` 和 `available_maps`。若 `rpc_connected=false`：检查 CARLA 进程日志是否完成初始化、端口 2000/2001、容器或远程主机的实际 IP、版本正确的 PythonAPI；若 `version_match=false`：换用与你 0.9.16 服务器匹配的 `carla` wheel/egg。**不要直接把 `doctor.dependencies.carla=true` 当作已连接**。网络不通可执行 `ss -ltn | grep ':2000'` 检查监听端口（需要服务器 shell）。

只有 RPC 成功且地图可用时再执行：

```bash
python -m caver.cli carla-run --host 127.0.0.1 --port 2000 \
  --town Town04 --step 0.05 --horizon 8 --out outputs/carla
cat outputs/carla/carla_summary.json
```

此处会实际设置同步世界、生成车辆、附加碰撞传感器，并在退出时尝试恢复世界设置和销毁 actor。**不能并发让其他脚本也驱动该 CARLA world**；若找不到四车连续道路或碰撞事件无效，保留报错和地图名称。CARLA 实验的 ego 控制与 frozen Python IDM 不完全一致，不能强行比较两者的碰撞数字。报告应检查 `paired_initial_state_differences`；其不为零时属于物理初始状态不完全配对。

## 3. 先跑 CBC 合成机制对照：推荐两档

先运行轻量级**流程验收**；不据此得出性能结论：

```bash
MODE=pilot bash scripts/cbc_synthetic_study.sh 2>&1 | tee outputs/cbc_synthetic_pilot.log
```

输出默认在 `outputs/cbc_synthetic_pilot/`：

```text
train_records.jsonl.gz        训练用源反应器预测和合成参考审计，非用户需上传的数据
 test_records.jsonl.gz        独立 seed 的合成参考审计，非需上传数据
cbc_full.pt                   完整候选条件源预测 + 物理风险头（高计算预算基线）
cbc_no_events.pt              移除事件特征的受控消融
cbc_anchor_full.pt            只复用一个 anchor + 风险残差预测
cbc_anchor_graph.pt           只复用一个 anchor + 严格过去→未来延迟图
cbc_study.json                小体积、成对 bootstrap 的 CBC 排序/风险对比
cbc_boundary_study.json       训练子集拟合、测试子集冻结的**非认证**经验复用筛选
cbc_gate.json                 样本量/参考 MC/研究结论证据充足性审计
rq_audit.json                 补充风险、事件裕度和来源诊断
```

**study** 扩大样本并对参考响应器 Monte Carlo 扫描：

```bash
MODE=study TRAIN_SCENES=100 TEST_SCENES=120 AUDIT_WORLDS=96 \
  CBC_EPOCHS=65 GRAPH_EPOCHS=20 \
  bash scripts/cbc_synthetic_study.sh \
  2>&1 | tee outputs/cbc_synthetic_study.log
```

`AUDIT_WORLDS` 表示每一参考反应器家族的独立世界数；默认 2 个家族。参考风险差的有限样本显著性取决于 **K=2×AUDIT_WORLDS**；若 `critical_pair_ref_sign_resolved` 太少，应通过事先指定的更大 K 和独立场景扩样改善，而不是看了 test 结果后调整 critical threshold。所有 synthetic-stress 分布以既有 posthoc 四车案例为中心，**只能估计该压力分布下的性能，不能估计自然驾驶风险事件发生率**。

参数可以用 `TRAIN_SCENES`、`TEST_SCENES`、`SOURCE_WORLDS`、`AUDIT_WORLDS`、`CBC_EPOCHS`、`GRAPH_EPOCHS`、`BOOTSTRAP`、`CBC_OUT` 覆盖。`MODE=pilot` 只作错误定位。

## 4. WOMD：不再输出超大训练 JSONL，且不下载额外的 tf.Example training

WOMD v1.3.1 官方 Scenario 包含 10 帧历史、1 帧当前、80 帧未来；**每个时刻的规划输入只应包含该时刻及其以前**。本版 `womd-prep` 仅调用 `with_future=False`，不把未来轨迹拷贝至 `Scene`。但当前 adapter 只提取每个原生窗口的 `current_time_index` 这个 planning instant，并**未**重建连续车辆控制轨迹/完整闭环时间序列；要覆盖每个 instant 的完整训练还需在后续作严格按时钟切片并以原始段/日志分组。

WOMD 官方同一 20 秒原段可能拆成重叠 9 秒窗口，所以**仅比对 scene_id 不足以宣称跨集合独立**。当前 train/calibration 以静态地图 lane IDs 生成启发式 group；validation 作为内部保留评估。它可减少同地图窗碰撞，但**没有真实 parent segment ID 保证**。如果要求顶级会议中严肃的无泄露结论，应补获取/重建可靠 log/segment ID，或采用原生分割并审计重叠时空轨迹指纹；已知不完整的验证结果不应声称无泄露证明。

直接执行：

```bash
MODE=pilot bash scripts/cbc_womd_pipeline.sh 2>&1 | tee outputs/cbc_womd_pilot.log
```

这条脚本依次：`pytest`/WOMD 预检、`womd-prep policy`、训练/校准/验证 `womd-prep corridor`、观察历史行为训练、指定源代理反应器、生成 train/cal/test 对照、强制划分审计、CBC 训练与消融、冻结测试审计以及证据门槛。

`womd_policy_train.jsonl.gz` 和 `womd_{train,calibration,test}_corridor.jsonl.gz` 每个文件压缩后**严格小于默认 45 MiB（约 47.2 MB）**。超过预算时停止抽样并在对应 `.manifest.json` 写 `size_budget`，不会继续输出无界大文件；压缩记录本身能继续直接训练，不要求展开为 JSONL。对 WOD `training` 的 `train` 与 `calibration` 通过分组选择，不从官方 validation 拿 calibration。

若实验需要更大样本：

```bash
MODE=study POLICY_SCENES=3000 TRAIN_SCENES=120 CAL_SCENES=40 TEST_SCENES=120 \
  MAX_SCAN=110000 MAX_SHARDS=380 PER_SHARD=550 \
  SOURCE_WORLDS=8 AUDIT_WORLDS=96 CBC_EPOCHS=65 GRAPH_EPOCHS=20 \
  bash scripts/cbc_womd_pipeline.sh \
  2>&1 | tee outputs/cbc_womd_study.log
```

这只是一条**研究代理模型**流水线：初始观测是真 WOMD，非 Ego action 的对照未来是由 `learned_logged`（WOMD 历史回归器）以及 `late_brake/conservative` 合成结构响应器仿真出来的。WOMD 真实日志中不存在未执行的 Early/Late 反事实标签。需要用两个独立外部实现的反应式模型、CARLA 真实跟随者模型、Waymax 支持的闭环系统或 ReactSim 原生 runner 来审计实际决策价值。

如果 `womd-prep` 报 `No eligible scenes exported`：打开 `outputs/cbc_womd_*/womd_*_corridor.jsonl.gz.manifest.json` 中 `scanned` / `skip_reasons`，先提高 `MAX_SCAN`,`MAX_SHARDS` 或检查 WOMD 地图和四车几何规则。**不能为了产生正面结果而事后根据测试标签挑选候选**。某些地区/车辆密度可能缺少符合构型的四车链，这本身是有意义的筛选覆盖率结果。

需要人工分步执行、优先导出很小的样本时：

```bash
python -m caver.cli womd-prep \
  --split training --partition train --purpose policy --limit 100 \
  --max-scan 10000 --max-shards 30 --per-shard 350 \
  --max-output-mb 5 --output outputs/womd_history_small.jsonl.gz

python -m caver.cli womd-prep \
  --split validation --partition test --purpose corridor --limit 10 \
  --max-scan 20000 --max-shards 75 --per-shard 400 \
  --max-output-mb 5 --output outputs/womd_corridor_small.jsonl.gz
```

**注意**：`womd-export` 是兼容旧版的原始导出接口，可能生成巨型 JSONL；现在的 `server_pipeline.sh` 已重定向为上面的 CBC-first、压缩且限额的方案。旧文件 `outputs/womd_training.jsonl` 和 `outputs/womd_validation.jsonl` 在确认新导出可复用后可自行移走或删除，模型不需要它们。

## 5. 怎么读 CBC 结果：重点不是轨迹 ADE

主指标在 `cbc_study.json`：

| 指标 | 学术问题 | 判定要求 |
|---|---|---|
| `matched_kinematic_proxy_scenes` | Early/Late 是否为同意图、近似任务、动力学合法的动作？ | 只有匹配样本进入主 ranking 统计；地图合法性另需证实 |
| `critical_pair_tests` / `critical_pair_ref_sign_resolved` | 是否有幅度达到预设阈值、且参考蒙特卡洛能可靠分辨符号的 pair 对？ | 解析不足要增 K/样本；不可把不确定符号当真标签 |
| `reference_resolved_sign_error_rate` | 风险顺序的符号是否反了？ | 主指标应优于 `anchor_copy`，最好也优于 `no_events` 与可计算预算对应的基线 |
| `critical_pair_underresolution_rate` | 模型是否把危险差别估计得几乎为零？ | 边界漏报应下降，不能靠整体高风险泛化掩盖 |
| `near_boundary_pair_risk_mae` / `far_boundary...` | 优势是否集中在 clearance 近边界？ | 报告分组覆盖率，避免阈值 posthoc 调参 |
| `downstream_union_reference_risk_mae` | 独立联合第三方碰撞事件是否准确？ | 不能用 pair risk 相加取代 joint event |
| `matched_ego_safe_safety_ranking_regret` | 错择候选的风险代价多大？ | 与 source/full 和 anchor-copy 比较；这里没含完整任务目标 |
| `source_contact_flip_missed_reference_fraction` | 源模型是否漏掉另一响应家族的接触反转？ | 漏检越低越好，但模型归因仅限所选反应器 |

**消融对照务必匹配计算预算**：

- `source`：各候选都跑独立源反应器，**昂贵但更强**的全量基线。
- `anchor_copy`：只运行 Early 源反应，把它直接沿用到 Late 和 reference，**便宜但经常无法区分候选**。
- `anchor_full` / `anchor_graph`：**单 anchor** + 候选 ego path/动作编码 + 风险头/严格延迟图；这是论文核心的**可复用预测成本约束**。
- `full`：每个 candidate 都运行源反应器再用 learned risk correction。
- `no_events`：以相同训练标签和结构移除事件时刻、事件有效性、停车裕度及传播路径特征。

`cbc_study.json` 同时输出相对 `source`、相对 `anchor_copy`、`full` 相对 `no_events` 的 scene-level bootstrap 区间。`cbc_gate.json` 的**预先声明**证据充足性条件：至少 100 测试场景、50 个匹配场景、30 个参考符号可分辨的关键候选 pair、128 个参考 world/action。没达到则 `INSUFFICIENT_REFERENCE_EVIDENCE`，不准写 CBC 胜出。即使 gate 出现 `DESCRIPTIVE_SIMULATOR_CONDITIONAL_SIGNAL_ONLY`，仍需检查区间是否支持改进、真实地图合法性以及外部闭环模型复现。

经验 guard `cbc_boundary_study.json` 在 train 开发数据选阈值，在独立 test 冻结评估，返回 `missed_flip_pairs`、`reuse_fraction`、`clopper_pearson_95pct_upper_miss_fraction`。由于尚无验证的动力学敏感度、干预扰动包络、所有 relevant guards/真实 swept geometry，上述经验筛选**绝不可**宣称满足论文可证明边界复用的条件。部署必须拒绝不充分认证、重新算 full response 或采用独立物理安全前缀。

## 6. 把回传给分析模型的内容压缩到很小

**不要上传** `*_records.jsonl.gz`、`womd_*_corridor.jsonl.gz`、`womd_policy_train.jsonl.gz`、`.pt`、CARLA 全部轨迹、原始 WOMD TFRecord、nuPlan DB。这些是本地训练用中间文件；`*.manifest.json` 保存大小、SHA-256、扫描/选择数与筛选跳过原因。

只回传**诊断包**：

```bash
python scripts/package_diagnostics.py \
  outputs/cbc_synthetic_pilot outputs/cbc_womd_pilot \
  --output outputs/CaVER_CBC_report_bundle.zip \
  --log outputs/cbc_womd_pilot.log
```

若只完成合成实验：

```bash
python scripts/package_diagnostics.py outputs/cbc_synthetic_pilot \
  --output outputs/CaVER_CBC_report_bundle.zip
```

白名单只收 `cbc_study.json`、`cbc_boundary_study.json`、`cbc_gate.json`、`rq_audit.json`、`split_audit.json`、CARLA check 和类似清单/少量日志；自动拒绝压缩后的原始训练样本和模型 checkpoint。默认 ZIP 总体上限 20 MiB，并且必须小于 50 MiB。配套 MANIFEST 带 checksum；上传这个 zip 即可在后续对照研究问题和调参方向进行分析。

## 7. nuPlan、Waymax、ReactSim：何时引入与执行指令

**nuPlan**：你现有 DB/map 可以作为下一阶段地图一致性与闭环基准，而不是在 CBC 可证伪数据不足时立即接入。先运行：

```bash
python -m caver.cli nuplan-check --output outputs/nuplan_inventory.json
DB=$(find /data0/senzeyu2/dataset/nuplan/data/cache/public_set_val -name '*.db' | head -n 1)
python -m caver.cli nuplan-check --db "$DB" --output outputs/nuplan_db_check.json
```

若已安装官方 nuPlan devkit：

```bash
python -m caver.cli nuplan-export --db "$DB" \
  --map-root /data0/senzeyu2/dataset/nuplan/maps \
  --limit 2 --output outputs/nuplan_observation_check.jsonl.gz
```

它输出的是 **历史观察和地图 API 元数据，不是可直接视为闭环的专家未来**；官方 expert route 可能经过未来状态重建，所以源码刻意没有作为规划输入。避免使用 `public_set_test` 做调参。

**Waymax**：当前主要读取 Scenario，不需要下载 tf.Example training；需要运行 Waymax 原生数据 loader/基线时按其文档下载对应 `uncompressed/tf_example/{training,validation}`，绝不能把两种格式当两个独立样本池。

**ReactSim-Bench**：等 CBC 压力测试至少形成可判断的模型条件结论后，再下载官方响应模型和行为数据（nuPlan 数据和地图已经存在）：

```bash
bash scripts/download_reactsim.sh
conda create -n reactsim_dp python=3.9 pip -y
conda activate reactsim_dp
bash scripts/reactsim_preprocess.sh
```

这些脚本下载官方仓库、HF 行为文件与 checkpoints，建立不覆盖原 DB/map 的链接；`reactsim_preprocess.sh` 在官方 3.9 环境执行其 `preprocess.sh`。它们**不等于**正式 ReactSim reactive rollout 或获得了 terminal certification，后者还需运行仓库的 benchmark/evaluation 并验证输出来源。官方说明：<https://github.com/Thinklab-SJTU/ReactSim-Bench/blob/main/docs/install_and_preprocess.md>。

## 8. CBC 阶段通过/失败之后，DSE 和 TIV 的研究优先级

若 `critical_pair_ref_sign_resolved` 不足：优先修候选匹配/基准场景筛选和参考反应器 K，不调 DSE 阈值；若 `anchor_graph` 输给 `anchor_copy`，先修 event transport、源模型偏差和延迟边，而不是加 DSE/TIV 掩盖风险排序失败；若 `full` 优于 `anchor_graph` 但 anchor 方法优势在预算上很明显，研究**verified guard 失败回退与计算性价比**；若所有代理 CBC 都只在合成 law 有效，则需 ReactSim/Waymax/CARLA 的独立外部验证，论文宣称保持模型条件性的范围。

CBC 至少显示跨种子/域的稳定正向趋势后，优先补 DSE：独立 calibration、整个 episode 的 union/coherent risk interval、有效物理前缀、`RISK-ADMISSIBLE` vs `DECISION-CERTIFIED` 及 Risk–Coverage、abstention、wrong-accepted rate 线；TIV 则独立验证来源、查询延迟持有动作、查询后更新的真实信息状态、**独立 terminal split** 及 matched budget regret。当前原型 DSE/TIV **不满足**论文冻结 adaptive query 后 end-to-end terminal calibration 与实体驾驶物理安全的全部假设。

## 9. 当前仍然缺少的“论文完整复现”清单

1. WOMD/nuPlan **有地图拓扑的合法候选**（目前只有几何筛选；`matched_kinematic_proxy` 不是法规验证）；完整场景中所有可接触车辆的保护集而非固定四车。
2. 完整 guard 枚举、驾驶策略支持范围、包含延迟与离散误差的**经独立验证**响应扰动/接触包络，以及模式分支重算；当前严格因果时延图是可训练代理残差网络，不是论文 Proposition 的物理证书。
3. 引入外部真实独立反应器作为 counterfactual **reference-law**、追踪与源模型不同的 agent policy family 和校准误差，而非只是 `delayed_idm` 参数扰动；WOMD 历史能训练策略但不能提供 off-log 潜在结果标签。
4. 真实执行时间的 rollout、独立 swept continuous collision / error bound、纯动力学/地图保持前缀安全查核。
5. DSE 从完整前缀/查询后信息状态进行 end-to-end 校准，以及有证据的选择释放风险/覆盖率界；TIV 的独立 query source、deadline、query 后 terminal recalibration。
6. `training_20s` 的重叠母片段审计、跨场景原始日志 ID、不同城市/道路密度的自然分布考察；修复后要重跑 heldout，不可后验挑选正样本。

**科学上诚实的阶段结论：** v2 是 CBC 的可执行“有证据门槛的零假设检验工程”，足以对当前机制提出否定/初步支持性证据；不是在官方闭环上完整复现 CaVER 全理论算法。报告图表需分别注记 `SYNTHETIC_SURROGATE`、`WOMD_OBSERVED_INITIALIZATION` 与 `EXTERNAL_REACTIVE_CLOSED_LOOP`，不得互换。
